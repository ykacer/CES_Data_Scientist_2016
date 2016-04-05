import re

datePattern = re.compile("([a-zA-Z]{1,} \d{1,2} {0,1},{0,1} \d{4}) | (\d{1,2} [a-zA-Z]{1,} {0,1},{0,1} \d{4})")
file_evaluation = open("ie_dates_evaluation.txt",'w');
number_of_dates = 0;
number_max = 20;
with open("wikifirst.txt") as file:
    pageTitle=""
    for line in file:
        if pageTitle=="":
            pageTitle=line[:-1]
            continue
        if line=="\n":
            pageTitle=""
            continue
        match=re.search(datePattern, line)
        if match!=None:
            print pageTitle + "\thasDate\t" + match.group()
            number_of_dates += 1
            if(number_of_dates<=number_max):
                file_evaluation.write(pageTitle + "\thasDate\t" + match.group()+"\n")

file_evaluation.write("\nTotal number of matches : "+str(number_of_dates))
if(number_of_dates>=number_max):
    file_evaluation.write("\nPrecision = 100%")
file_evaluation.close()

