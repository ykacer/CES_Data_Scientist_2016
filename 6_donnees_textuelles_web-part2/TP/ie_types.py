import re

# firstly : clean wikifirst.txt
clean0 = open('wikifirst.txt','r').read()
cleanerPattern1 = re.compile("( |-)[A-Z]\w+(ese|an|sh|ern)") # put off nationality adjectif
cleanerPattern2 = re.compile(" (([0-9]+(st|nd|rd|th))|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth|tenth|eleventh|twelfth)") # put off numeral adjectif
cleanerPattern3 = re.compile(" (the )?(bigg?|larg|form|small|most|less|short|out|good|best)(er|est)?") # put off superlatifs
cleanerPattern4 = re.compile(" (or|and|of)") # put off linking words
cleanerPattern5 = re.compile(" (some|like|one|kind|what|sort|part|name|those|the|we|you\every)") # put off pronouns
cleanerPattern6 = re.compile(" \w+ful") # put of common adjectif like *ful
cleanerPattern7 = re.compile(" \w+ed (into|to|for) \w+") # put of verbal describer (like 'used to describe','dedicated for doing')
clean1 = re.sub(cleanerPattern1,"",clean0)
clean2 = re.sub(cleanerPattern2,"",clean1)
clean3 = re.sub(cleanerPattern3,"",clean2)
clean4 = re.sub(cleanerPattern4,"",clean3)
clean5 = re.sub(cleanerPattern5,"",clean4)
clean6 = re.sub(cleanerPattern6,"",clean5)
clean7 = re.sub(cleanerPattern7,"",clean6)


file_clean = open('wikifirst_clean.txt','w')
file_clean.write(clean7);
file_clean.close()



typePattern = re.compile("(\w+( is| are| was| were) the \w+)|(\w+  ?(is|are|was|were)( an?| the)? (\w|-| )+ ?(.|,))")
file_evaluation = open("ie_types_evaluation.txt",'w');
number_of_types = 0;
objectPattern = re.compile("\w+ ?(.|,)$")
with open("wikifirst_clean.txt") as file:
    pageTitle=""
    for line in file:
        if pageTitle=="":
            pageTitle=line[:-1]
            continue
        if line=="\n":
            pageTitle=""
            continue
        match=re.search(typePattern, line)
        if match!=None:
	    obj = re.search(objectPattern, match.group(0))
	    if obj!=None:
           	print pageTitle + "\thasType\t" + obj.group(0)
           	number_of_types += 1
            	file_evaluation.write(pageTitle + "\thasType\t" + obj.group(0)+"\n")

file_evaluation.write("\nTotal number of matches : "+str(number_of_types))
file_evaluation.close()

