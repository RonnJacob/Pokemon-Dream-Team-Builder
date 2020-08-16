   # Pokémon : Let's find the dream team!

#### Loading the necessary libraries/code

```python
from IPython.display import Image
from IPython.display import YouTubeVideo
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd

from vega_datasets import data


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

#### Loading the dataset (aka. the Pokédex)

```python
pokedex = pd.read_csv('pokemon.csv')
pokedex.head()
```

As we can see, we've got 801 instances/pokemon with about 41 attributes for each
pokemon.

```python
pokedex.shape
```

```python
pokedex.name
```

```python
pokedex.columns = pokedex.columns.str.lower()
```

```python
pokedex["id"] = pokedex.index + 1
```

```python
pokedex
```

## Cleaning the data

```python
pokedex.drop_duplicates('name', keep='first', inplace=True)
```

```python
pokedex.shape
```

#### We have attributes Type 1 and Type 2. All Pokémon don't necessarily have 2
types. For eg :  Charizard can be of Fire and Flying type, however Squirtle is
of Water type. The data for Type 2 for squirtle is NaN. So this needs to be set
to None.

```python
pokedex.loc[pokedex['name'] == 'Squirtle']
```

```python
pokedex['type2'].fillna(value='None', inplace=True)
```

```python
pokedex.loc[pokedex['name'] == 'Squirtle']
```

#### We're going to sum up the net effect of attacks of a pokemon on all the
pokemon types by summing all the against_* attributes

```python
pokedex['against_all'] = pokedex['against_bug'] + pokedex['against_dark'] + pokedex['against_dragon'] + pokedex['against_electric'] 

pokedex['against_all'] += pokedex['against_fairy'] + pokedex['against_fight'] + pokedex['against_fire'] + pokedex['against_flying']

pokedex['against_all'] += pokedex['against_ghost'] + pokedex['against_grass'] + pokedex['against_ground'] + pokedex['against_ice']

pokedex['against_all'] += pokedex['against_normal'] + pokedex['against_poison'] + pokedex['against_psychic'] + pokedex['against_rock']

pokedex['against_all'] += pokedex['against_steel'] + pokedex['against_water']

pokedex
```

#### We can divide the set of attributes and main data into two sections - based
on ID.

The first section would be called the pokedata which would give the details
about the pokemon - this would be the name, types, generation of pokemon and
whether the pokemon is legendary or not.

The section of data would include the statistics of the pokemon -

```python

pokedata = pokedex[['id', 'name', 'type1', 'type2', 'generation', 'is_legendary', 'capture_rate']]
pokestats = pd.merge(pokedex, pokedata,on='id').loc[:, ['id', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'height_m', 'weight_kg', 'against_all','base_total']]
```

```python
pokedata.head(10)
```

```python
pokestats.head()
```

## Figuring out the data through visualizations

```python
sns.set(style="darkgrid")
ax = sns.countplot(x="type1", data=pokedata)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.figure(figsize=(50, 23))
plt.tight_layout()
plt.show()
```

##### As we can see, the greater number of pokemon types are the water types,
which is followed by normal, grass and bug respectively.

Now that we know the distribution of the types of pokemon for each of the types.
Let's delve into how many types of pokemon of each type exist. Keep in mind
that, unless specified, type would also imply that we're talking about the type1
of a pokemon.

```python
pd.value_counts(pokedata['type1']).plot.bar()
```

We can see below the counts of each type of pokemon for each of the 7
generations. We're able to clearly see the counts on the heatmap. It's helpful
when the list is annotated.

```python
sns.heatmap(
    pokedata.groupby(['type1', 'generation']).size().unstack(),
    linewidths=1,
    annot=True
);
```

### Ordering the pokemon by attack

```python
df1 = pokestats.sort_values(['attack','sp_attack'],ascending=False).groupby('attack')
df1 = df1.apply(pd.DataFrame)

attack_ordered_pokemon = pd.merge(df1, pokedata, on='id').head(50)

attack_ordered_pokemon.head(10)

```

We can see that when the pokemon is grouped by the attack statistics, which
include the special attack statistics - we can see that the top 5 Pokemon that
has the highest attack would be Heracross, Kartana, Rayquaza, Groudon, and
Garchomp.

I wondered what the most strongest attack pokemon types would be. So I sampled
200 pokemon by taking the top 200 pokemon with high attack values, and checked
to see the type count of the pokemon.

```python
pd.value_counts(attack_ordered_pokemon['type1']).plot.bar()
```

Here we can see that, the pokemon with the highest attack features are bug types
which is followed by water, dragon and rock

### Ordering the pokemon by defense

```python
df1 = pokestats.sort_values(['defense','sp_defense'],ascending=False).groupby('defense')
df1 = df1.apply(pd.DataFrame)

defense_ordered_pokemon = pd.merge(df1, pokedata, on='id').head(50)

defense_ordered_pokemon.head(10)


```

We can see that when the pokemon is grouped by the defense statistics, which
include the special defense statistics - we can see that the top 5 Pokemon that
has the highest defense would be Shuckle, Steelix, Aggron, Regirock, Avalugg.

I wondered what the most strongest defense pokemon types would be. So I sampled
200 pokemon by taking the top 200 pokemon with high defense values, and checked
to see the type count of the pokemon.

```python
pd.value_counts(defense_ordered_pokemon['type1']).plot.bar()
```

Here we can see that, the pokemon with the highest attack features are rock and
steel. This is pretty obvious seeing as to how thick shells, and rough skin can
be good defense against any type.

### Ordering Pokemon by capture rate.

It's been quite a while since I've played the games, and something that I've
wanted to do was to figure out how capture_rate plays an important role in
starting off and slowly building your team towards a strong dream team. You tend
to capture pokemon early and work with them throughout the game. So a list of
pokemon that could be on the team from the start could be the ones with higher
capture rates.

```python
pokedata = pokedex[['id', 'name', 'type1', 'type2', 'generation', 'is_legendary', 'capture_rate']]
pokedata['capture_rate'].astype('str').astype('int')
pokedata.sort_values(['is_legendary', 'capture_rate'], ascending=[True, False])
```

We can see how Caterpies, Pidgeys, Rattatas are the Pokemons with the higher
capture rate. This is pretty obvious from the games as well. They are much more
common when you tend to wander off into the wild grass.

### Other observations

Another interesting observation to make would be to count the number of
legendary Pokemon in the list of Pokemon. Legendary Pokemon are Pokemon that are
harder to catch, and if we're going to make the dream team, we should think
about having a dream team which is realistic. This involves having Pokemon which
aren't hard to catch, and Pokemon which have a good capture rate.

```python
 print(pokedata[pokedata['is_legendary'] > 0].count())
```

In the list of 801 Pokemon, we have around 70 legendary Pokemon which shows that
they're only 8% of the total number of Pokemons, which proves how rare they
might be.

```python
pokedata[pokedata['is_legendary'] > 0].head(3)
```

```python
pokedata[pokedata['is_legendary'] == 0].head(3)
```

We can also see how the capture rate for the legendary Pokemon vary from non
legendary Pokemon to legendary Pokemon.

#### Correlation of height and weight with base stats (attack, defense, etc)

```python
alt.Chart(pokestats).mark_bar().encode(
    x='attack',
    y='height_m'
)
```

```python
alt.Chart(pokestats).mark_bar().encode(
    x='attack',
    y='weight_kg'
)
```

```python
alt.Chart(pokestats).mark_bar().encode(
    x='defense',
    y='weight_kg'
)
```

```python
alt.Chart(pokestats).mark_bar().encode(
    x='defense',
    y='height_m'
)
```

### Let's get down to some very interesting results!

```python
Image(url='https://media1.tenor.com/images/6808aa5544b6867b20a689660b2e80e2/tenor.gif')
```

We're going to see if we can find out the strongest type of pokemon! How
exciting!

We need to first figure out the most common types of Pokemon types, and this
involves delving into the duality of Pokemon types - this would mean that the
Pokemon could be of two types.

```python
sns.factorplot(
    y='type1',
    data=pokedata,
    kind='count',
    order=pokedata['type1'].value_counts().index,
    color='red'
).set_axis_labels('# of Pokemon', 'Type 1')


sns.factorplot(
    y='type2',
    data=pokedata,
    kind='count',
    order=pokedata['type1'].value_counts().index,
    color='blue'
).set_axis_labels('# of Pokemon', 'Type 1')


```

We can see the distribution of each types seperately. However, what we need is
to find out the intersection count to find out what the most common dual type
Pokemon is.

```python
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(
    pokedata.groupby(['type1', 'type2']).size().unstack(),
    linewidths=2,
    annot=True,
    annot_kws={'size': 14},
    cmap="BuPu"
);
```

```python
dualtype = pokedata[pokedata['type2'] != 'None']
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(dualtype.groupby(['type1', 'type2']).size().unstack(),linewidths=2,annot=True,annot_kws={'size': 14},
    cmap="Reds");
```

So the two heatmaps above are different in a way that the former heatmap does
not omit single type Pokemon. Due to this reason, the top 5 common Pokemon types
are:

1. Normal/None
2. Water/None
3. Grass/None
4. Psychic/None
5. Grass/None

When we start to include only dual type Pokemon however, we start to see some
interesting observations. The latter heatmap shows us these results.

1. Normal/Flying
2. Grass/Poison
3. Bug/Flying
4. Bug/Poison
5. Water/Ground

We already have an attribute which would be a good general indicator of strong
pokemon, this would be the base_total. This would be the sum of the HP, attack,
defense, speed, special attack and special defense values. We could also add the
against_all to create a new field to see if the attacks against other types
plays a role in determining the strong pokemon.

```python
pokestats['grand_total'] = pokestats['base_total'] + pokestats['against_all']

powerful_pokemon = pd.merge(pokedata, pokestats, on='id').sort_values('grand_total', ascending=False)
powerful_pokemon.head(10)
```

We can see the top 10 most seemingly very powerful pokemon out of which only one
isn't a legendary Pokemon. Tyranitar which is the most evolved form of
Pupitar/Larvitar. We can proceed to take about the first 50 Pokemon which is 6%
and check to see what the distribution of types of Pokemon would be.

```python
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(
    powerful_pokemon.head(50).groupby(['type1', 'type2']).size().unstack(),
    linewidths=2,
    annot=True,
    annot_kws={'size': 14},
    cmap="Reds"
);
```

The most common powerful types of Pokemon would be

1. Normal/None
2. Water/None
3. Dragon/Flying

But this includes all the legendary Pokemon. Catching legendary Pokemon in the
games aren't cake walk, so we would need to exclude these Pokemon and see what
the non-legendary types of Pokemon would be, and proceed to narrow down the
types of Pokemon for the same.

```python
powerful_non_legendary_pokemon = powerful_pokemon[powerful_pokemon['is_legendary'] == 0]
powerful_non_legendary_pokemon.head(10)
```

Great. This looks like a promising result.


Moving forward, we have another parameter to consider which was mentioned
earlier - which was the capture_rate. Let's sort the powerful non legendary
pokemon by capture rate and grand total of stats.

```python
capture_ordered_nl_pokemon = powerful_non_legendary_pokemon.sort_values(['capture_rate','grand_total'],ascending=False).groupby('capture_rate')
capture_ordered_nl_pokemon = capture_ordered_nl_pokemon.apply(pd.DataFrame)

capture_ordered_nl_pokemon.head(10)
```

```python
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(capture_ordered_nl_pokemon.head(50).groupby(['type1', 'type2']).size().unstack(),linewidths=2,annot=True,
    annot_kws={'size': 14},cmap="Reds");
```

The common types of powerful pokemon are water, normal and ice types.

Therefore, when we consider a dream team, we could go with two approaches -

1. The first approach would be to choose Pokemon based on the capture rate, and
form a team from the availability of the Pokemon.

2. The second approach would be to choose Pokemon based on sheer strength, with
their base stats total.

The following could be the 2 variations of the dream team of 6 Pokémon

```python
capture_ordered_nl_pokemon = powerful_non_legendary_pokemon.sort_values(['capture_rate','grand_total'],ascending=False).groupby('grand_total')
capture_ordered_nl_pokemon = capture_ordered_nl_pokemon.apply(pd.DataFrame)

capture_ordered_nl_pokemon.head(6)['name']
```

```python
capture_ordered_nl_pokemon = powerful_non_legendary_pokemon.sort_values(['grand_total','capture_rate'],ascending=False).groupby('grand_total')
capture_ordered_nl_pokemon = capture_ordered_nl_pokemon.apply(pd.DataFrame)

capture_ordered_nl_pokemon.head(6)['name']
```

The above teams looks pretty balanced to succeed in any of the Nintendo Pokemon
games released, and they would be fairly easy to capture in the games.

## Building a legendary Pokémon classifier

Let's try to see the distribution of legendary pokemon in the dataset. From the
games, they are pretty rare Pokémon.

```python
fig = plt.figure(figsize=(7,7))

legendary = pokedata[pokedata['is_legendary'] == True]
non_legendary = pokedata[pokedata['is_legendary'] == False]

legendary_distr = [legendary['name'].count(), non_legendary['name'].count()]
legendary_pie = plt.pie(legendary_distr, labels= ['Legendary', 'Non Legendary'],autopct ='%1.1f%%', shadow = True,
                 colors = ["red", "blue"], startangle = 45, explode=(0, 0.1))
```

As we can see, Legendary Pokémon  are rare. This proves to be a problem since
there isn't enough data to exploit, which would mean that we would need to
analyze whether there would be correlations between stats, gender, capture_rate
or other attributes to determine whether a pokemon is legendary.

```python
# We're going to see if there is a correlation between attack, defense stats to determine whether the Pokemon is legendary.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



X = pokestats[['attack']]
Y = pokestats[['defense']]

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color = 'yellow')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()
```

```python
sns.boxplot(x=pokedex['is_legendary'], y=pokedex['attack'])
```

```python
sns.boxplot(x=pokedex['is_legendary'], y=pokedex['defense'])
```

```python
isLegendary = pd.get_dummies(pokedex['is_legendary'], drop_first=True)
legendary_poke = pokedex[['base_total', 'capture_rate']]
legendary_poke = pd.concat([legendary_poke, isLegendary], axis=1)
legendary_poke.columns = ['base_total', 'capture_rate', 'is_legendary']
legendary_poke.head(10)
```

```python
## Creating labels for legendary and non legendary pokemon

X_leg = legendary_poke.drop('is_legendary', axis = 1)
Y_leg = legendary_poke['is_legendary']
```

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold

X_leg_train, X_leg_test, Y_leg_train, Y_leg_test = train_test_split(X_leg, Y_leg, test_size=0.3) 


#Fitting the model to the training data.
LR = LogisticRegression()
LR.fit(X_leg_train, Y_leg_train)
```

```python
pred = LR.predict(X_leg_test)
print(classification_report(Y_leg_test, pred))
```

We can see how the following classification plays out with decision trees.

```python
kfold = KFold(n_splits=500, random_state=10)

clf = tree.DecisionTreeClassifier().fit(X_leg,Y_leg)

result = cross_val_score(clf, X_leg,Y_leg, cv=kfold, scoring='accuracy')

print(result.mean())
```

We can see that a decision tree classifier plays out better in determining
whether a pokemon is legendary with the attributes of base stats and capture
rate.

To conclude this project, we've done a remarkable job in analyzing this dataset.

1. We've figured out what a dream team could look like?
2. What kinds of pokemon are the strongest?
3. Whether we can tell if a pokemon is legendary or not?

This project has been really helpful in terms of understanding how different
visualizations, algorithms can be really helpful in deducing and analyzing data.

```python
Image(url='https://media1.tenor.com/images/36391cec14724d3ea5486c5d0b5e2b80/tenor.gif')

```

```python
YouTubeVideo('rg6CiPI6h2g')
```

# The End!

```python

```
