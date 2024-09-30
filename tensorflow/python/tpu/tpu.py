# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================

"""Library of TPU helper functions."""

import collections
import enum
import typing
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union

from absl import logging
import numpy as np

from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export


ops.NotDifferentiable("TPUReplicatedInput")

# Operations that indicate some error in the users graph, e.g. a placeholder
# that's introduced outside of the infeed.
_DENYLISTED_OPS = set([
    "Placeholder",
])

# XLA doesn't currently support reading of intermediate tensors, thus some ops
# are not supported.
_UNSUPPORTED_OPS = set([
    "AudioSummary",
    "AudioSummaryV2",
    "HistogramSummary",
    "ImageSummary",
    "MergeSummary",
    "Print",
    "ScalarSummary",
    "TensorSummary",
    "TensorSummaryV2",
    ])

# Ops which can be safely pruned from XLA compile if they have no consumers.
#  These ops should also have no inputs.
_UNCONNECTED_OPS_TO_PRUNE = set(["Placeholder", "VarHandleOp"])

_MAX_WARNING_LINES = 5

_TPU_REPLICATE_ATTR = "_tpu_replicate"
_POST_DEVICE_REWRITE_ATTR = "_post_device_rewrite"
_TPU_COMPILATION_STATUS_ATTR = "_tpu_compilation_status"
_OUTSIDE_COMPILATION_ATTR = "_xla_outside_compilation"
_PIVOT_FOR_CLUSTER = "_pivot_for_cluster"


core = tpu_name_util.core


def _tpu_system_device_name(job: Optional[Text]) -> Text:
  """Returns the device name for the TPU_SYSTEM device of `job`."""
  if job is None:
    return "/device:TPU_SYSTEM:0"
  else:
    return "/job:%s/device:TPU_SYSTEM:0" % job


@tf_export(v1=["tpu.initialize_system"])
def initialize_system(
    embedding_config: Optional[embedding_pb2.TPUEmbeddingConfiguration] = None,
    job: Optional[Text] = None,
    compilation_failure_closes_chips: bool = True,
    tpu_cancellation_closes_chips: Optional[bool] = None,
) -> core_types.Tensor:
  """Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, a `TPUEmbeddingConfiguration` proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
    compilation_failure_closes_chips: Set the configuration whether
      we want to close TPU chips when there is a compilation failure.
    tpu_cancellation_closes_chips: Set the configuration whether
      we want to close TPU chips when a TPU execution is cancelled. If the value
      is None, the behavior will be determined by the command line flag
      `tpu_cancellation_closes_chips` for the TPU worker. WARNING: this argument
      only applies to TFRT TPU runtime.
  Returns:
    A serialized `TopologyProto` that describes the TPU system. Note:
      the topology must be evaluated using `Session.run` before it can be used.
  """
  config_string = ("" if embedding_config is None else
                   embedding_config.SerializeToString())

  # The enum is defined in core/tpu/kernels/tpu_execute_op_options.h.
  tpu_cancellation_closes_chips_enum = 0
  if tpu_cancellation_closes_chips is not None:
    if tpu_cancellation_closes_chips:
      tpu_cancellation_closes_chips_enum = 1
    else:
      tpu_cancellation_closes_chips_enum = 2

  with ops.device(_tpu_system_device_name(job)):
    topology = tpu_ops.configure_distributed_tpu(
        compilation_failure_closes_chips=compilation_failure_closes_chips,
        tpu_cancellation_closes_chips=tpu_cancellation_closes_chips_enum,
    )

    if embedding_config is None:
      return topology

    # This set of control dependencies is needed as this function is expected to
    # return an op which will return the topology when executed, but we need to
    # call the embedding initialization op between initializing the TPU and
    # returning the topology.
    with ops.control_dependencies([topology]):
      embedding_init = tpu_ops.configure_tpu_embedding(config=config_string)
    with ops.control_dependencies([embedding_init]):
      return array_ops.identity(topology, name="tpu_init_identity")


def initialize_system_for_tpu_embedding(
    embedding_config: embedding_pb2.TPUEmbeddingConfiguration,
    job: Optional[Text] = None,
) -> ops.Operation:
  """Initializes a distributed TPU Embedding system for use with TensorFlow.

  The following two are equivalent:
  1. initialize_system() with embedding_config.
  2. initialize_system() without embedding_config, then
     initialize_system_for_tpu_embedding().
  initialize_system() should not be called with embedding_config if
  initialize_system_for_tpu_embedding() is meant to be called later.

  Args:
    embedding_config: a `TPUEmbeddingConfiguration` proto describing the desired
      configuration of the hardware embedding lookup tables.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.

  Returns:
    A no-op.
  """
  config_string = embedding_config.SerializeToString()
  with ops.device(_tpu_system_device_name(job)):
    return tpu_ops.configure_tpu_embedding(config=config_string)


@tf_export(v1=["tpu.shutdown_system"])
def shutdown_system(job: Optional[Text] = None) -> ops.Operation:
  """Shuts down a running a distributed TPU system.

  Args:
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be shutdown. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
  """
  with ops.device(_tpu_system_device_name(job)):
    shutdown_distributed_tpu = tpu_ops.shutdown_distributed_tpu()
  return shutdown_distributed_tpu


def _enclosing_tpu_context_and_graph() -> Tuple[Any, Any]:
  """Returns the TPUReplicateContext and its associated graph."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, TPUReplicateContext):
        return context_, graph
      context_ = context_.outer_context
    graph = getattr(graph, "outer_graph", None)
  raise ValueError("get_replicated_var_handle() called without "
                   "TPUReplicateContext. This shouldn't happen. Please file "
                   "a bug.")


def is_tpu_strategy(strategy: Any) -> bool:
  is_tpu_strat = lambda k: k.__name__.startswith("TPUStrategy")
  clz = strategy.__class__
  return is_tpu_strat(clz) or any(map(is_tpu_strat, clz.__bases__))


def _enclosing_tpu_device_assignment(
) -> Optional[device_assignment_lib.DeviceAssignment]:
  if not distribution_strategy_context.has_strategy():
    return None
  strategy = distribution_strategy_context.get_strategy()
  if not is_tpu_strategy(strategy):
    return None
  return strategy.extended._device_assignment  # pylint: disable=protected-access


@auto_control_deps.register_acd_resource_resolver
def tpu_replicated_input_resolver(
    op: ops.Operation,
    resource_reads: object_identity.ObjectIdentitySet,
    resource_writes: object_identity.ObjectIdentitySet) -> bool:
  """Replaces TPUReplicatedInput outputs with its inputs in resource_inputs."""
  # Ignore TPUReplicatedInput for ACD purposes since we will be directly adding
  # control deps on the replicated inputs.
  if op.type == "TPUReplicatedInput":
    if resource_reads or resource_writes:
      resource_reads.clear()
      resource_writes.clear()
      return True
    else:
      return False
  # Replace tensors in `resource_inputs` which are outputs of TPUReplicatedInput
  # with the actual replicated inputs. This allows ACD to correct add control
  # deps when there are multiple calls to `run` in a
  # `tf.function`.
  def replace_with_unreplicated_resources(resource_inputs):
    """Replaces handles in `resource_inputs` with their unreplicated inputs."""
    to_remove = []
    to_add = []
    for resource in resource_inputs:
      if resource.op.type == "TPUReplicatedInput":
        to_remove.append(resource)
        to_add.extend(resource.op.inputs)
    for t in to_remove:
      resource_inputs.discard(t)
    resource_inputs.update(to_add)
    return to_add or to_remove

  return bool(replace_with_unreplicated_resources(resource_reads) or
              replace_with_unreplicated_resources(resource_writes))


class TPUReplicateContext(control_flow_ops.XLAControlFlowContext):
  """A `ControlFlowContext` for nodes inside a TPU computation.

  The primary role of `TPUReplicateContext` is to mark operators inside a
  tpu.replicate() computation with the attribute "_tpu_replicate=XYZ", where XYZ
  is a unique name.

  We use a `ControlFlowContext` to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a tpu.replicate() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the replicated computation.
  """

  def __init__(self, name: Text, num_replicas: int, pivot: ops.Operation):
    """Builds a new TPUReplicateContext.

    Args:
      name: a unique name for the context, used to populate the `_tpu_replicate`
        attribute.
      num_replicas: an integer that gives the number of replicas for the
        computation.
      pivot: a pivot node. Nodes in the TPUReplicateContext that do not have any
        inputs will have a control dependency on the pivot node. This ensures
        that nodes are correctly included in any enclosing control flow
        contexts.
    """
    super(TPUReplicateContext, self).__init__()
    self._num_replicas = num_replicas
    self._outer_device_function_stack = None
    self._oc_dev_fn_stack = None
    self._outside_compilation_cluster = None
    self._outside_compilation_v2_context = None
    self._outside_compilation_counter = 0
    self._in_gradient_colocation = None
    self._gradient_colocation_stack = []
    self._host_compute_core = []
    self._name = name
    self._name_as_bytes = compat.as_bytes(name)
    self._tpu_relicate_attr_buf = c_api_util.ScopedTFBuffer(
        attr_value_pb2.AttrValue(s=self._name_as_bytes).SerializeToString())
    self._unsupported_ops = []
    self._pivot = pivot
    self._replicated_vars = {}

  def get_replicated_var_handle(self,
                                name: Text,
                                handle_id: Text,
                                vars_: Union[List[core_types.Tensor],
                                             List[variables.Variable]],
                                is_mirrored: bool = False,
                                is_packed: bool = False) -> core_types.Tensor:
    """Returns a variable handle for replicated TPU variable 'var'.

    This is a method used by an experimental replicated variable implementation
    and is not intended as a public API.

    Args:
      name: The common name of the variable.
      handle_id: Unique ID of the variable handle, used as the cache key.
      vars_: The replicated TPU variables or handles.
      is_mirrored: Whether the variables are mirrored, which guarantees the
        values in each replica are always the same.
      is_packed: Whether the replicated variables are packed into one variable.

    Returns:
      The handle of the TPU replicated input node.
    """
    device_assignment = _enclosing_tpu_device_assignment()
    # We don't need to put device assignment as part of the replicated_vars key
    # because each TPUReplicateContext will only have one device assignment.
    handle = self._replicated_vars.get(handle_id)
    if handle is not None:
      return handle

    if device_assignment is not None and not is_packed:
      # Find a variable copy for each replica in the device assignment.
      # Note that the order of devices for replicas for the variable and the
      # device assignment might not match.
      job_name = pydev.DeviceSpec.from_string(vars_[0].device).job
      devices_to_vars = {device_util.canonicalize(v.device): v for v in vars_}
      replicated_vars = []
      for replica_id in range(device_assignment.num_replicas):
        for logical_core in range(device_assignment.num_cores_per_replica):
          device = device_util.canonicalize(
              device_assignment.tpu_device(
                  replica=replica_id, logical_core=logical_core, job=job_name))
          if device in devices_to_vars:
            replicated_vars.append(devices_to_vars[device])
            break
        else:
          raise ValueError(
              "Failed to find a variable on any device in replica {} for "
              "current device assignment".format(replica_id))
    else:
      replicated_vars = vars_

    # Builds a TPUReplicatedInput node for the variable, if one does not already
    # exist. The TPUReplicatedInput node must belong to the enclosing
    # control-flow scope of the TPUReplicateContext.
    # TODO(phawkins): consider changing the contract of the TPU encapsulation
    # so the TPUReplicatedInput nodes go inside the TPUReplicateContext scope
    # instead.

    _, graph = _enclosing_tpu_context_and_graph()
    with graph.as_default():
      # If replicated_vars are variables, get the handles. Note that this can be
      # done inside TPUReplicateContext because replicated_vars.handle may
      # create new ops.
      if isinstance(replicated_vars[0], variables.Variable):
        replicated_vars = [v.handle for v in replicated_vars]
      # pylint: disable=protected-access
      saved_context = graph._get_control_flow_context()
      graph._set_control_flow_context(self.outer_context)
      handle = tpu_ops.tpu_replicated_input(replicated_vars,
                                            name=name + "/handle",
                                            is_mirrored_variable=is_mirrored,
                                            is_packed=is_packed)
      graph._set_control_flow_context(saved_context)
      # pylint: enable=protected-access
    self._replicated_vars[handle_id] = handle
    return handle

  def report_unsupported_operations(self) -> None:
    if self._unsupported_ops:
      op_str = "\n".join("  %s (%s)" % (op.type, op.name)
                         for op in self._unsupported_ops[:_MAX_WARNING_LINES])
      logging.warning("%d unsupported operations found: \n%s",
                      len(self._unsupported_ops), op_str)
      if len(self._unsupported_ops) > _MAX_WARNING_LINES:
        logging.warning("... and %d more" %
                        (len(self._unsupported_ops) - _MAX_WARNING_LINES))

  def EnterGradientColocation(self, op: ops.Operation, gradient_uid: Text):
    if op is not None:
      if ops.get_default_graph()._control_flow_context is None:  # pylint: disable=protected-access
        # If we are in TF 2 functions (control flow V2 functions, or
        # tf.function()), we need to attach _xla_outside_compilation attribute
        # directly because we are not in TPUReplicateContext.
        try:
          outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode("ascii")
        except ValueError:
          # The attr was not present: do nothing.
          return
        parts = outside_attr.split(".")
        cluster = parts[0] + "." + gradient_uid
        self._outside_compilation_v2_context = OutsideCompilationV2Context(
            cluster)
        self._outside_compilation_v2_context.Enter()
        return
      self._gradient_colocation_stack.append(op)
      if not self._outside_compilation_cluster:
        try:
          outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode("ascii")
          if self._in_gradient_colocation:
            raise NotImplementedError(
                "Cannot nest gradient colocation operations outside compilation"
            )
          if gradient_uid == "__unsupported__":
            raise NotImplementedError(
                "No gradient_uid calling gradient within outside_compilation")
          # When we take the gradient of an op X in an outside_compilation
          # cluster C in a forward computation we would like to put the ops
          # corresponding to the gradient of X into a new outside_compilation
          # cluster C'. However, if we take the gradient of X twice, the second
          # one should get yet another new outside_compilation cluster C''.
          #
          # The mechanism we adopt is to use a 'root_cluster' which is the
          # cluster that X was in before we took gradients, and a 'gradient_uid'
          # which is different for every invocation of gradients, and put the
          # gradient of X in cluster 'root_cluster.gradient_uid'.
          #
          # When taking a gradient of a gradient, some ops will be colocated
          # with Op in the forward pass (e.g., cluster root_cluster) and some in
          # the backward pass (e.g., cluster root_cluster.initial_gradient_uid).
          # We need all of the grad-of-grad ops to be in the same cluster to
          # avoid cyclic dependencies between clusters. We adopt a heuristic
          # that puts any op clustered with root_cluster.<xxx> in
          # root_cluster.gradient_uid, even if xxx was initial_gradient_uid.
          self._in_gradient_colocation = op
          parts = outside_attr.split(".")
          cluster = parts[0] + "." + gradient_uid
          self._EnterOutsideCompilationScope(cluster=cluster)
        except ValueError:
          # The attr was not present: do nothing.
          pass

  def ExitGradientColocation(self, op: ops.Operation, gradient_uid: Text):
    if op is not None:
      if ops.get_default_graph()._control_flow_context is None:  # pylint: disable=protected-access
        # Inside a TF2 tf.function or control flow graph and `op` was not
        # t do
    tion or control flow graph  nter(resent: do nothing.
 ion or h and m