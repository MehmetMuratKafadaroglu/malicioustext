from text_classify import *
from timeit import timeit

def test_and_dump():
    sexual_content_filters = [BayesianSexualContentFilter(), SexualContentFilter()]
    for filter in sexual_content_filters:
        filter.train()
        filter.dump()
        print(filter("Show the tits", "The weather is so nice today"))
        print(filter.accuracy)

    racism_filters = [BayesianRacismFilter(), RacismFilter()]
    for racism_filter in racism_filters:
        racism_filter.train()
        racism_filter.dump()
        print(racism_filter("Jews and gypsies are the worst, blacks are bit better", "The weather is so nice today"))
        print(racism_filter.accuracy)

    sexism_filters= [BayesianSexismFilter(), SexismFilter()]
    for sexism_filter in sexism_filters:
        sexism_filter.train()
        sexism_filter.dump()
        print(sexism_filter("Place of the woman is kitchen, men are superior to women", "The weather is so nice today"))
        print(sexism_filter.accuracy)

    cbullying_filters = [CyberBullyingFilter(),BayesianCyberBullyingFilter(), ]
    for cyberbullying_filter in cbullying_filters:
        cyberbullying_filter.train()
        cyberbullying_filter.dump()
        print(cyberbullying_filter("You look ugly, go kill yourself", "The weather is so nice today"))
        print(cyberbullying_filter.accuracy)

def test_from_load():
    sexual_content_filters = [BayesianSexualContentFilter(), SexualContentFilter()]
    racism_filters = [BayesianRacismFilter(), RacismFilter()]
    sexism_filters= [BayesianSexismFilter(), SexismFilter()]
    cbullying_filters = [CyberBullyingFilter(),BayesianCyberBullyingFilter(), ]

    for filter in sexual_content_filters:
        filter.load()
        print(filter("Show the tits", "The weather is so nice today"))

    for filter in racism_filters:
        filter.load()
        print(filter("Jews and gypsies are the worst, blacks are bit better", "The weather is so nice today"))

    for filter in sexism_filters:
        filter.load()
        print(filter("Place of the woman is kitchen, men are superior to women", "The weather is so nice today"))
    
    for filter in cbullying_filters:
        filter.load()
        print(filter("You look ugly, go kill yourself", "The weather is so nice today"))


"""
0.8819112627986349 Bayesian Sexual
0.9312855517633675 Random Forest Sexual

0.9182903834066625 Bayesian Racism
0.98 Random Forest Racism

0.882172131147541 Bayesian Sexism
0.9375 Random Forest Sexism

0.839832285115304 Bayesian Bullying
0.8517819706498951 Random Forest Bullying
"""
test_and_dump()