from bs4 import BeautifulSoup

read_files = open('listfiles.txt', 'r')
file_names = read_files.readlines()

write_file = open('articles.txt', 'a')
for file in file_names:
    file = file.strip() 
    try:
        f = open(file, 'r')
        soup = BeautifulSoup(f, 'html.parser')
        txt = soup.get_text()
        print_txt = ""
        for word in txt.split():
            # if word != "\n":
            print_txt += (word)
            print_txt += " "
        write_file.write(print_txt)
        write_file.write('\n')
    except:
        print("error in", file)
        continue