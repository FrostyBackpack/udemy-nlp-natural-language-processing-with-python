import spacy

# language library
# efficient as running on top of is preloaded into this library
nlp = spacy.load("en_core_web_sm")

# parse string into separate components as tokens (each word becomes a token)
doc = nlp(u"Tesla is looking at buying U.S. startup for $6 million")

for token in doc:
  print(token.text, token.pos_, token.dep_)

print(nlp.pipe_names)

doc2 = nlp(u"Tesla isn't      looking into startups anymore.")

for token in doc2:
  print(token.text, token.pos_, token.dep_)