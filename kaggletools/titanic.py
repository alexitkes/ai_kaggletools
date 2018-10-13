"""
Functions anlyzing "Titanic" data set, the training set for beginners
in machine learning. See `https://www.kaggle.com/c/titanic` for more
info.

Public functions defined here:

*   extract_title

Classed sefined here:

*   TicketCounter
*   CabinCounter
*   FamilyCorrector
"""

import warnings

def extract_title(data, titles=None):
    """
    Create a title column meaning whether the `data.Name` column
    contains Mr, Mrs, Miss, etc.title. Returns a Series of
    integers indicating the titles. Default mapping is the following.

    *   0 means Mr
    *   1 means Mrs
    *   2 means Miss
    *   3 means Master
    *   4 means rare titles like Dr, Sir, etc.

    This mapping may be customized using the `titles` argument
    (see below).

    Parameters
    ----------
    data : pandas.DataFrame
        The data frame that must contain the 'Name' column containing
        the 'Mr', 'Mrs', etc. title in addition to the name itself.

    titles : list, default ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
        The list of interesting titles. It must contain the Mr, Mrs,
        Miss, Master and Rare strings and may contain Dr, Military
        and Royal. For every passenger, the returned column will
        contain the integer meaning index of this passenger's title
        in the list.

    Returns
    -------
    pandas.Series
        The list of integers of range 0-4 (or `0` to `len (titles) - 1` if
        `titles` argument given) indicating the titles taken from the name
        column. The series index will be same with that of data argument.

    Raises
    ------
    Exception :
        The Mr, Mrs, Miss and Master title not allowed to return.

    Issues
    ------
    *   If `Rare` title not given as interesting, it may be a good idea
        to use 'Mr', 'Mrs', 'Miss', or 'Master' titles depending on
        gender and age.
    """
    if titles is None:
        titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
    else:
        warnings.warn("Using non-default title list is still an experimental feature and may change without deprecation")
    if 'Mr' not in titles or 'Mrs' not in titles or 'Miss' not in titles or 'Master' not in titles:
        raise Exception("Must be able to return Mr, Mrs, Miss and Master titles.")
    def _mapper(x):
        if x in ["Mr"]:
            return titles.index("Mr")
        elif x in ["Mrs", "Mme"]:
            return titles.index("Mrs")
        elif x in ["Miss", "Ms", "Mlle"]:
            return titles.index("Miss")
        elif x in ["Master"]:
            return titles.index("Master")
        elif x in ["Dr"] and "Dr" in titles:
            return titles.index("Dr")
        elif x in ["Capt", "Major", "Col"] and "Military" in titles:
            return titles.index("Military")
        elif x in ["Sir", "Count", "Countess"] and "Royal" in titles:
            return titles.index("Royal")
        else:
            return titles.index("Rare")
    return data.Name.str.extract("([A-Za-z]+)\\.", expand=False).map(_mapper)

#
# The functions used to calculate survival of the passenger's family members.
#
class TicketCounter(object):
    def __init__(self, data, simplified=False, fill_if_not_any_survived=False):
        self.data = data
        self.simplified = simplified
        self.fill_if_not_any_survived = fill_if_not_any_survived
    
    def fill_ticket_rates(self):
        counts = {}
        surv = {}
        died = {}
        for t in self.data.Ticket.unique():
            counts[t] = self.data[self.data.Ticket == t].PassengerId.count()
            surv[t] = self.data[(self.data.Ticket == t) & (self.data.Survived == 1)].Survived.count()
            died[t] = self.data[(self.data.Ticket == t) & (self.data.Survived == 0)].Survived.count()
        self.data["TicketCount"] = self.data.Ticket.map(lambda x: counts[x])
        self.data["TicketRate"] = np.NaN
        for i in self.data.index:
            ticket = self.data.loc[i, "Ticket"]
            pid = self.data.loc[i, "PassengerId"]
            if self.simplified:
                if self.fill_if_not_any_survived:
                    if self.data.loc[(self.data.Ticket == ticket) & (self.data.PassengerId != pid)].Survived.max() == 1.0:
                        self.data.loc[i, "TicketRate"] = 1.0
                    elif self.data.loc[(self.data.Ticket == ticket) & (self.data.PassengerId != pid)].Survived.min() == 0.0:
                        self.data.loc[i, "TicketRate"] = 0.0
                else:
                    if self.data.loc[(self.data.Ticket == ticket) & (self.data.PassengerId != pid)].Survived.min() == 1.0:
                        self.data.loc[i, "TicketRate"] = 1.0
                    elif self.data.loc[(self.data.Ticket == ticket) & (self.data.PassengerId != pid)].Survived.max() == 0.0:
                        self.data.loc[i, "TicketRate"] = 0.0
            else:
                s = surv[self.data.loc[i, "Ticket"]]
                d = died[self.data.loc[i, "Ticket"]]
                if np.isfinite(self.data.loc[i].Survived) and (self.data.loc[i].Survived == 1):
                    s -= 1
                if np.isfinite(self.data.loc[i].Survived) and (self.data.loc[i].Survived == 0):
                    d -= 1
                if s + d:
                    self.data.loc[i, "TicketRate"] = float(s) / float(s + d)
        if self.simplified:
            self.data["TicketRate"] = self.data["TicketRate"].fillna(0.5)
        else:
            for c in self.data.Pclass.unique():
                self.data.loc[self.data.Pclass == c, "TicketRate"] = self.data.loc[self.data.Pclass == c, "TicketRate"].fillna(self.data.loc[(self.data.Pclass == c), "Survived"].mean())
        

class CabinCounter(object):
    def __init__(self, data, filler=None, simplified=False):
        self.data = data
        self.simplified = simplified
        if filler is None:
            if simplified:
                self.filler = pd.Series(0.5, index=data.index)
            else:
                self.filler = pd.Series(np.NaN, index=data.index)
                for c in self.data.Pclass.unique():
                    self.filler.loc[self.data.Pclass == c] = self.data.loc[self.data.Pclass == c, "Survived"].mean()
        else:
            self.filler = filler
    
    def fill_cabin_rates(self):
        counts = {}
        surv = {}
        died = {}
        for c in self.data.loc[~self.data.Cabin.isna(), "Cabin"].unique():
            counts[c] = self.data[self.data.Cabin == c].PassengerId.count()
            surv[c] = self.data[(self.data.Cabin == c) & (self.data.Survived == 1)].Survived.count()
            died[c] = self.data[(self.data.Cabin == c) & (self.data.Survived == 0)].Survived.count()
        self.data.loc[self.data.Cabin.isna(), "CabinCount"] = 0
        self.data.loc[~self.data.Cabin.isna(), "CabinCount"] = self. data.loc[~self.data.Cabin.isna(), "Cabin"].map(lambda x: counts[x])
        self.data["CabinRate"] = np.NaN
        for i in self.data.index:
            if not self.data.loc[i, "Cabin"] in counts:
                continue
            cabin = self.data.loc[i, "Cabin"]
            pid = self.data.loc[i, "PassengerId"]
            if self.simplified:
                if self.data.loc[(self.data.Cabin == cabin) & (self.data.PassengerId != pid)].Survived.min() == 1.0:
                    self.data.loc[i, "CabinRate"] = 1.0
                elif self.data.loc[(self.data.Cabin == cabin) & (self.data.PassengerId != pid)].Survived.max() == 0.0:
                    self.data.loc[i, "CabinRate"] = 0.0
            else:
                s = surv[cabin]
                d = died[cabin]
                if np.isfinite(self.data.loc[i].Survived) and (self.data.loc[i].Survived == 1):
                    s -= 1
                if np.isfinite(self.data.loc[i].Survived) and (self.data.loc[i].Survived == 0):
                    d -= 1
                if s + d:
                    self.data.loc[i, "CabinRate"] = float(s) / float(s + d)
        self.data.loc[self.data.CabinRate.isna(), "CabinRate"] = self.filler.loc[self.data.CabinRate.isna()]

class FamilyCorrector(object):
    """
    Corrects predictions of the first model (that only analyaze
    features of individual passengers) with the second model that
    analyzes family relations.
    """
    def _new_family(self, i):
        idx = len(self.families)
        self.families.loc[idx,'Pclass'] = self.data.loc[i, "Pclass"]
        self.families.loc[idx,'Embarked'] = self.data.loc[i, "Embarked"]
        self.families.loc[idx,'Lastname'] = self.data.loc[i, "Lastname"]
        self.families.loc[idx,'Size'] = 0
        self.families.loc[idx,'Id'] = idx
        return idx

    def _find_family(self, i):
        fams = self.families.loc[(self.families.Pclass == self.data.loc[i, "Pclass"]) & (self.families.Embarked == self.data.loc[i, "Embarked"])]
        if not len(fams):
            return self._new_family(i)
        exact = fams.loc[fams.Lastname == self.data.loc[i, "Lastname"]]
        if len(exact):
            return exact.iloc[0].Id
        if self.data.loc[i, "SecondaryLastname"] == np.NaN:
            return self._new_family(i)
        secondary = fams.loc[fams.Lastname == self.data.loc[i, "SecondaryLastname"]]
        if len(exact):
            return exact.iloc[0].Id
        else:
            return self._new_family(i)

    def _fill_family_ids(self):
        self.families = pd.DataFrame(columns=['Pclass', 'Embarked', 'Lastname', 'Id', 'Size'])
        self.data["Family"] = np.NaN
        if self.simplified and self.use_fare:
            self.data["Family"] = np.NaN
            fid = 0
            for i in self.data.index:
                f = self.data.loc[i].Fare
                l = self.data.loc[i].Lastname
                if np.isfinite(self.data.loc[i].Family):
                    continue
                self.data.loc[(self.data.Fare == f) & (self.data.Lastname == l), "Family"] = fid
                fid += 1
        else:
            for i in self.data.index:
                # Leave family ID NAN if no family.
                if self.data.loc[i, "SibSp"] == 0 and self.data.loc[i, "Parch"] == 0:
                    continue
                self.data.loc[i, "Family"] = self._find_family(i)
                f = self.data.loc[i, "Family"]
                self.families.loc[self.families.Id == f, "Size"] = self.families.loc[self.families.Id == f, "Size"] + 1
            for f in self.families.index:
                if self.families.loc[f].Size != 1:
                    continue
                fid = self.families.loc[f].Id
                pid = self.data.loc[self.data.Family == fid, "PassengerId"].iloc[0]
                pclass = self.data.loc[self.data.Family == fid, "Pclass"].iloc[0]
                emb = self.data.loc[self.data.Family == fid, "Embarked"].iloc[0]
                second_name = self.data.loc[self.data.Family == fid, "SecondaryLastname"].iloc[0]
                if second_name == np.NaN:
                    continue
                fams = self.families.loc[self.families.Lastname == second_name]
                for fam in fams.index:
                    fam_id = fams.loc[fam].Id
                    c = self.data.loc[self.data.Family == fid, "Family"].count()
                    self.families.loc[self.families.Id == fam_id, "Size"] = self.families.loc[self.families.Id == fam_id, "Size"] + c
                    self.data.loc[self.data.Family == fid, "Family"] = fam_id
                    self.families.loc[f, "Size"] = 0
                    break
                if self.families.loc[f, "Size"] == 0:
                    continue
                if self.data.loc[self.data.PassengerId == pid, "SibSp"].iloc[0] == 0:
                    continue
                if self.data.loc[self.data.PassengerId == pid, "Sex"].iloc[0] != 1:
                    continue
                sisters = self.data.loc[(self.data.SecondaryLastname == second_name) &
                                        (self.data.Sex == 1) &
                                        (self.data.Pclass == pclass) &
                                        (self.data.Embarked == emb) &
                                        (self.data.PassengerId != pid)]
                if len(sisters) > 0:
                    fam_id = sisters.iloc[0].Family
                    c = self.data.loc[self.data.Family == fid, "Family"].count()
                    self.families.loc[self.families.Id == fam_id, "Size"] = self.families.loc[self.families.Id == fam_id, "Size"] + c
                    self.data.loc[data.Family == fid, "Family"] = fam_id
                    self.families.loc[f, "Size"] = 0
    
    def __init__(self, data, filler=None, simplified=False, use_fare=False, fill_if_not_any_survived=False):
        self.data = data
        self.simplified = simplified
        self.use_fare = use_fare
        self.fill_if_not_any_survived = fill_if_not_any_survived
        if 'Lastname' not in list(self.data.columns):
            self.data['Lastname'] = self.data['Name'].apply(lambda x: str.split(x, ",")[0])
        if 'SecondaryLastname' not in list(self.data.columns):
            self.data['SecondaryLastname'] = self.data.Name.str.extract("([A-Za-z'-]+)\\)$", expand=False)
            self.data['SecondaryLastname'] = self.data['SecondaryLastname'].fillna(self.data.Lastname.str.extract("^[A-Za-z]+-([A-Za-z]+)$", expand=False))
        if filler is None:
            if simplified:
                self.filler = pd.Series(0.5, index=data.index)
            else:
                self.filler = pd.Series(np.NaN, index=data.index)
                for c in self.data.Pclass.unique():
                    self.filler.loc[self.data.Pclass == c] = self.data.loc[self.data.Pclass == c, "Survived"].mean()
        else:
            self.filler = filler
    
    def fill_family_rates(self):
        self._fill_family_ids()
        self.data["FamilyRate"] = np.NaN
        self.data["ChildRate"] = np.NaN
        self.data["FemaleRate"] = np.NaN
        self.data["MaleRate"] = np.NaN
        for i in self.data.index:
            if self.data.loc[i, "Family"] == np.NaN:
                continue
            fam = self.data.loc[i, "Family"]
            pid = self.data.loc[i, "PassengerId"]
            if self.data.loc[self.data.Family == fam, "PassengerId"].count() == 1:
                continue
            if self.simplified:
                if self.fill_if_not_any_survived:
                    if self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid)].Survived.max() == 1.0:
                        self.data.loc[i, "FamilyRate"] = 1.0
                    elif self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid)].Survived.min() == 0.0:
                        self.data.loc[i, "FamilyRate"] = 0.0
                else:
                    if self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid)].Survived.min() == 1.0:
                        self.data.loc[i, "FamilyRate"] = 1.0
                    elif self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid)].Survived.max() == 0.0:
                        self.data.loc[i, "FamilyRate"] = 0.0
                if self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age <= 15)].Survived.min() == 1.0:
                    self.data.loc[i, "ChildRate"] = 1.0
                elif self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age <= 15)].Survived.max() == 0.0:
                    self.data.loc[i, "ChildRate"] = 0.0
                if self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 1)].Survived.min() == 1.0:
                    self.data.loc[i, "FemaleRate"] = 1.0
                elif self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 1)].Survived.max() == 0.0:
                    self.data.loc[i, "FemaleRate"] = 0.0
                if self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 0)].Survived.min() == 1.0:
                    self.data.loc[i, "MaleRate"] = 1.0
                elif self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 0)].Survived.max() == 0.0:
                    self.data.loc[i, "MaleRate"] = 0.0
            else:
                self.data.loc[i, "FamilyRate"] = self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid)].Survived.mean()
                self.data.loc[i, "ChildRate"] = self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age <= 15)].Survived.mean()
                self.data.loc[i, "FemaleRate"] = self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 1)].Survived.mean()
                self.data.loc[i, "MaleRate"] = self.data.loc[(self.data.Family == fam) & (self.data.PassengerId != pid) & (self.data.Age > 15) & (self.data.Sex == 0)].Survived.mean()
        self.data.loc[self.data.FamilyRate.isna(), "FamilyRate"] = self.filler.loc[self.data.FamilyRate.isna()]
        if self.simplified and self.fill_if_not_any_survived:
            self.data.loc[self.data.FamilyRate < 0.75, "FamilyRate"] = self.filler.loc[self.data.FamilyRate < 0.75]
        if self.simplified:
            self.data["ChildRate"] = self.data["ChildRate"].fillna(0.5)
            self.data["FemaleRate"] = self.data["FemaleRate"].fillna(0.5)
            self.data["MaleRate"] = self.data["MaleRate"].fillna(0.5)
        else:
            for c in self.data.Pclass.unique():
                self.data.loc[self.data.Pclass == c, "ChildRate"] = self.data.loc[self.data.Pclass == c, "ChildRate"].fillna(self.data.loc[(self.data.Pclass == c) & (self.data.Age <= 15), "Survived"].mean())
                self.data.loc[self.data.Pclass == c, "FemaleRate"] = self.data.loc[self.data.Pclass == c, "FemaleRate"].fillna(self.data.loc[(self.data.Pclass == c) & (self.data.Age > 15) & (self.data.Sex == 1), "Survived"].mean())
                self.data.loc[self.data.Pclass == c, "MaleRate"] = self.data.loc[self.data.Pclass == c, "MaleRate"].fillna(self.data.loc[(self.data.Pclass == c) & (self.data.Age > 15) & (self.data.Sex == 0), "Survived"].mean())
        self.data.loc[self.data.Age <= 15, "OwnRate"] = self.data.loc[self.data.Age <= 15, "ChildRate"]
        self.data.loc[(self.data.Age > 15) & (self.data.Sex == 1), "OwnRate"] = self.data.loc[(self.data.Age > 15) & (self.data.Sex == 1), "FemaleRate"]
        self.data.loc[(self.data.Age > 15) & (self.data.Sex == 0), "OwnRate"] = self.data.loc[(self.data.Age > 15) & (self.data.Sex == 0), "MaleRate"]
        self.data["NumOlder"] = 0
        self.data["NumYounger"] = 0
        self.data["NumParents"] = 0
        for i in self.data.index:
            fid = self.data.loc[i, "Family"]
            age = self.data.loc[i, "Age"]
            if fid == np.NaN:
                continue
            self.data.loc[i, "NumOlder"] = self.data.loc[(self.data.Family == fid) & (self.data.Age > age)].PassengerId.count()
            self.data.loc[i, "NumYounger"] = self.data.loc[(self.data.Family == fid) & (self.data.Age < age)].PassengerId.count()
            self.data.loc[i, "NumParents"] = min(self.data.loc[i, "Parch"], self.data.loc[(self.data.Family == fid) & (self.data.Age > age + 20)].PassengerId.count())
        self.data["AgeRank"] = self.data.NumYounger - self.data.NumOlder
        self.data["AgeRank"] = pd.cut(self.data.AgeRank, [-25, -2.5, 0, 2.5, 25], labels=False)
        self.data["NumChildren"] = self.data.Parch - self.data.NumParents
