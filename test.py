import spacy
import sys
reload(sys)
sys.setdefaultencoding('utf8')
nlp=spacy.load('en_core_web_md')
document=nlp(u"what are we going to do about this I don't know")

print([token.text for token in document])

