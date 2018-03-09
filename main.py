import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
import urllib.request
import bs4 as bs

class Cuisine_Recommendation():
    def get_cuisines_from_website(self):
        source = urllib.request.urlopen('https://tasty.co/').read()
        soup = bs.BeautifulSoup(source, 'lxml')
        cuisines = []
        self.all_cuisines = []
        for url in soup.find_all('a'):
            cuisines.append(url.get('href'))
        compilations = list(filter(lambda cmp: 'https://tasty.co/compilation/' in cmp, cuisines))
        cuisines = list(filter(lambda csn: 'https://tasty.co/recipe/' in csn, cuisines))
        cookbook = {'cuisines': cuisines, 'compilations': compilations}
        for cuisine in cookbook['compilations']:
            cookbook_source = urllib.request.urlopen(cuisine).read()
            soup = bs.BeautifulSoup(cookbook_source, 'lxml')
            temp_cuisines = []
            for url in soup.find_all('a'):
                temp_cuisines.append(url.get('href'))
            self.all_cuisines.extend(list(filter(lambda csn: 'https://tasty.co/recipe/' in csn, temp_cuisines)))
            self.all_cuisines.extend(cookbook['cuisines'])
        self.all_cuisines = set(self.all_cuisines)
        self.all_cuisines = list(self.all_cuisines)

    def write_ingredients_to_csv(self):
        my_file = open('data.csv', 'w')
        my_file.write('ingredients,url\n')
        for cuisine in self.all_cuisines:
            source = urllib.request.urlopen(cuisine).read()
            soup = bs.BeautifulSoup(source, 'lxml')
            ingredients = soup.findAll("li", {"class": "xs-mb1 xs-mt0"})
            ingredients = [str(i) for i in ingredients]
            pure_ing = []
            for ing in ingredients:
                matched = re.search(r"\>(.*?)\<", ing)
                if 'boneless' in matched.group(1) and 'skinless' in matched.group(1):
                    matched = matched.group(1).strip()
                else:
                    matched = matched.group(1).strip().split(',')[0]
                matched = ''.join([i for i in matched if not i.isnumeric()]).strip()
                words = r'lbs?|cups?|tablespoons?|teaspoons?|softened?|strips?|packets?|heads?|of\b|fillets?\b'
                words_rm = r'small?|large?|slices?|shredded?\b|sheets?|cans?\b|\d ?g\b|oz\b|boiling\b|pts?\b'
                words_rm_2 = r'diced\b|chopped\b|fresh\b|boxs?\b|sprigs?\b|warm\b|warmed\b|grated\b|boneless\b|skinless\b'
                other = r'[\d,;#\(\)\[\]\.]'
                out = re.sub(words, '', matched)
                out = re.sub(words_rm, '', out)
                out = re.sub(words_rm_2, '', out)
                out = re.sub(other, '', out)
                matched = out.replace('%', '').strip()
                matched = ' '.join(matched.split())
                pure_ing.append(matched)
                pure_ing = set(pure_ing)
                pure_ing = list(filter(None, pure_ing))
            my_file.write('"{}",{}\n'.format(",".join(pure_ing), cuisine))
        my_file.close()

    def preprocessing_data(self):
        data = pd.read_csv('data.csv', delimiter=',', encoding='latin-1')
        self.ingredients = data['ingredients']
        cuisine = data['url']
        vectorizer = CountVectorizer()
        self.X = vectorizer.fit_transform(self.ingredients).toarray()
        self.y = np.array(cuisine)

    def model(self, X_train, y_train, ingredients_tk):
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        vectorizer = CountVectorizer()
        vectorizer.fit(self.ingredients)
        ingredients_tk = vectorizer.transform([ingredients_tk]).toarray()
        probs = clf.predict_proba(ingredients_tk)
        recommendations = sorted(zip(clf.classes_, probs[0]), key=lambda x: x[1])[-3:]
        return reversed(recommendations)


if __name__ == '__main__':
    meal = Cuisine_Recommendation()
    meal.get_cuisines_from_website()
    meal.write_ingredients_to_csv()
    meal.preprocessing_data()
    ingredients_taken = input("Which ingredients do you have? ")
    output = meal.model(meal.X, meal.y, ingredients_taken)
    print("\nRecommended cuisines:")
    for i in output:
        print(i[0])
