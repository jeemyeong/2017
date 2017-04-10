'''
Modeling users, interactions and items from
the recsys challenge 2017.

by Daniel Kohlsdorf
'''

class User:

    def __init__(self, title, clevel, indus, disc, country, region, premium):
        self.title   = title
        self.clevel  = clevel
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region
        self.premium = premium

class Item:

    def __init__(self, title, clevel, indus, disc, country, region, is_payed):
        self.title   = title
        self.clevel  = clevel
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region
        self.is_payed  = is_payed

class Interaction:

    def __init__(self, user, item, interaction_type0, interaction_type1, interaction_type2, interaction_type3, interaction_type4, interaction_type5):
        self.user = user
        self.item = item
        self.interaction_type0 = interaction_type0
        self.interaction_type1 = interaction_type1
        self.interaction_type2 = interaction_type2
        self.interaction_type3 = interaction_type3
        self.interaction_type4 = interaction_type4
        self.interaction_type5 = interaction_type5
        self.encoded = None

    def title_match(self):
        return float(len(set(self.user.title).intersection(set(self.item.title))))

    def clevel_match(self):
        if self.user.clevel == self.item.clevel:
            return 1.0
        else:
            return 0.0

    def indus_match(self):
        if self.user.indus == self.item.indus:
            return 1.0
        else:
            return 0.0

    def discipline_match(self):
        if self.user.disc == self.item.disc:
            return 2.0
        else:
            return 0.0

    def country_match(self):
        if self.user.country == self.item.country:
            return 1.0
        else:
            return 0.0

    def region_match(self):
        if self.user.region == self.item.region:
            return 1.0
        else:
            return 0.0

    def premium(self):
        if self.user.premium == 1:
            return 1.0
        else:
            return 0.0

    def features(self):
        return [
            self.title_match(), self.clevel_match(), self.indus_match(),
            self.discipline_match(), self.country_match(), self.region_match(),
            self.premium()
        ]

    def addEncodeData(self,encodedData):
        self.encoded = encodedData
        print(encodedData)

    def label(self):
        score = 0.0
        # score += self.interaction_type0 * 0.3
        # score += self.interaction_type1 * 0.33
        # score += self.interaction_type2 * 0.45
        # score += self.interaction_type3 * 0.45
        # score += self.interaction_type4 * -0.6
        # score += self.interaction_type5 * 0.9
        if self.interaction_type5:
            score +=0.9
        elif self.interaction_type4:
            score +=0
        elif self.interaction_type3:
            score +=0.45
        elif self.interaction_type2:
            score +=0.45
        elif self.interaction_type1:
            score +=0.33
        else:
            score +=0.3
        return score
