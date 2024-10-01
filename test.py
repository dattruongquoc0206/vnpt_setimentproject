import argparse
from functions import *

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-a", "--Output", help = "add path to record folder")

# Read arguments from command line
args = parser.parse_args()

folder_path = f"{args.Output}"
print(folder_path)


con = sqlite3.connect("database.db")
cur = con.cursor()

if os.path.exists(folder_path):
    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # remove noise file before processing
        if get_size_kb(file_path) > 1:        
        # Only process files, not subdirectories
            if os.path.isfile(file_path):
                _, sr, channel, _, _, data = get_wav_informations(file_path)
                # print(f"{file_name}: {file_size}, {sr}, {channel}, {duration}, {silence_ratio}, {data}")
                cur.execute(f"""INSERT INTO wav VALUES ("{file_name}", "{file_path}", {sr}, {channel}, "{data}")""")
                # cur.execute(f"""INSERT INTO wav_feature VALUES ("{file_name}", "{file_size}", {duration}, {silence_ratio})""")

                con.commit()



con.close()