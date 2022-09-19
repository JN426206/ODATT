# Narzędzie do usuwana plików na podstawie innego katalogu zawierającego te same pliki (mogą być inne rozszerzenia).
## W przypadku usunięcia plików z katalogu źródłowego zostaną one również usunięte z katalogu docelowego - po to został stowrzony ten skrypt.
## Czyli narzędzie synchronizuje katalog do czyszczenia z katalogiem źródłowym.
# Przykładowe użycie (usunięcie zrzuconych klatek w katalogu "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/frames" na podstawie źródłowego katalogu "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/pred_frames"
# python tools/clear_directory.py "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/pred_frames" "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/frames"
# Przykładowe użycie (usunięcie plików tekstowych z adnotacjami w katalogu "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/frames_labels" na podstawie źródłowego katalogu "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/pred_frames"
# python tools/clear_directory.py "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/pred_frames" "2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich/frames_labels"
import glob
import os
import sys

def scan(target_to_clear, source):
    files = [os.path.basename(x) for x in glob.glob(source+"/*", recursive=True)]
    files_to_clear = [".".join(os.path.basename(x).split(".")[:-1]) for x in glob.glob(target_to_clear + "/*", recursive=True)]
    numoffiles = len(files)
    print("Number of files: {}".format(numoffiles))
    found = 0
    notfound = 0
    for file in files:
        if ".".join(file.split(".")[:-1]) in files_to_clear:
            found += 1
            # print("Found: {}/{}".format(found,numoffiles))
        else:
            notfound += 1
            os.remove(os.path.join(source, file))
            print("File removed: {}".format(os.path.join(source, file)))
    print("Not found and removed: {}".format(notfound))
    print("Found: {}/{}".format(found,numoffiles))

if __name__ == "__main__":
    #print(len(sys.argv))
    if len(sys.argv)<2:
        print("Podaj jako 1 argument katalog źródłowy a jako 2 katalog przeszukiwany w którym zostaną usunięte pliki!")
        exit(0)

    scan(sys.argv[1],sys.argv[2])