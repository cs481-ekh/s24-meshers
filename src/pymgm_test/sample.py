# python program to add punctuation to sentences

def add_punct(sentence, punctuation):
    if punctuation=="exclaimation":
        sentence += "!"
    elif punctuation=="period":
        sentence += "."
    elif punctuation=="question":
        sentence += "?"
    return sentence


#sentence = add_punct("Did you bake cake yesterday", "question")

#print(sentence)
