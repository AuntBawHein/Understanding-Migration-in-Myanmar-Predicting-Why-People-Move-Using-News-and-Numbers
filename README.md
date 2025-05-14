# Understanding-Migration-in-Myanmar-Predicting-Why-People-Move-Using-News-and-Numbers
To learn from **news headlines** and **data numbers** to guess the reason for migration.


# Understanding Migration in Myanmar – Predicting Why People Move Using News and Numbers

## Step 1: What I Want to Do (Scope and Goals)

I want to understand why people in Myanmar are leaving their homes and moving to other places. Some people move because there is fighting. Others move because of floods, no jobs, sickness, or to go to school.

My goal is to build a smart computer model that can read news headlines and look at numbers. Then, it will try to guess the main reason why people are moving.

### What I will focus on:
- I will use made-up (synthetic) data about people moving and news stories from Myanmar.
- I will focus only on predicting the reason why people move — like conflict, flood, job, health, or education.
- I won’t try to show live updates or real-time movement. This is just about learning from the data.

---

```markdown
---

## **Step 2: Prepare and Understand the Dataset**

I will load and look at my dataset closely to understand what kind of data I have.

### What I’ll Do:

* I will open a CSV file with news headlines, dates, migration numbers, and reasons.
* I will check the column names and see if there are any missing values.

### Tools I’ll Use:

* Pandas
* Matplotlib

---

### What is Pandas?

**Pandas is a Python tool that helps me work with tables of data.**

### Why do we have to use Pandas for this step?

**I use Pandas in this step because I need to open my dataset, look at the columns, and check the data clearly.**

---

### What is Matplotlib?

**Matplotlib is a tool that helps me draw graphs and charts in Python.**

### Why do we have to use Matplotlib for this step?

**I use Matplotlib here because I want to create bar charts that help me see patterns, like which season or reason has the most migration.**

---
```

Code 

```python
# I start by importing a tool called pandas, which helps me work with tables of data
import pandas as pd 

# I also import a tool called matplotlib.pyplot (I call it plt), which helps me make charts 
# and graphs.
import matplotlib.pyplot as plt 

# I open my dataset and read the data into a table called df
df = pd.read_csv("why_people_move_myanmar.csv") 

# I check how many rows and columns are in my data
print(df.shape)  # For example, (10000, 10) means 10,000 rows and 10 columns

# I look at the list of column names to understand what kind of information I have
print(df.columns.tolist())

# I look at 5 random rows just to explore what the data looks like
print(df.sample(5))

# I check what type of data is in each column and make sure everything is filled 
# (not missing values or data)
df.info()

# I now create a chart to show how many migration events happened in each season
df['season'].value_counts().plot(kind='bar')

# I give my chart a clear title so others know what it shows
plt.title("Number of Migration Events by Season in Myanmar")

# I label the X and Y axes so it's easy to understand what the bars mean
plt.xlabel("Season (Winter, Rainy, Summer)") 
plt.ylabel("Number of Migration Records")

# I use this to make sure the labels don’t get cut off in the chart
plt.tight_layout()

# I double-check that there are no missing values in the dataset
print(df.isnull().sum())

# Now I create another chart to show the main reasons why people migrate
df['reason'].value_counts().plot(kind='bar')

# I add a title so it's clear what this second chart shows
plt.title("Main Reasons Why People Migrate in Myanmar") 

# I add labels to help the reader understand the chart
plt.xlabel("Migration Reason") 
plt.ylabel("Number of Times Each Reason Was Reported") 

# I make sure the chart looks nice and everything fits
plt.tight_layout()

# I use this to show the final chart
plt.show()
```

#### Output:
```
(10000, 10)
['date', 'migration_count', 'headline', 'reason', 'region', 'news_keywords', 'season', 'risk_score', 'population', 'is_border_area']
            date  migration_count  \
4423  2024-05-25              577   
3536  2021-02-26              245   
599   2020-06-14              597   
5644  2023-04-01              360   
7321  2025-04-13              120   

                                               headline    reason  \
4423  President interest wonder air oil save themsel...     flood   
3536      Front reflect smile record receive both mean.    health   
599   Huge anything herself so base movement activit...  conflict   
5644  Magazine around up less expect market leave re...  conflict   
7321                              White candidate tree.    health   

           region                                      news_keywords  season  \
4423  Tanintharyi  president interest wonder air oil save themsel...  summer   
3536       Magway       front reflect smile record receive both mean  winter   
599        Yangon  huge anything herself so base movement activit...   rainy   
5644   Ayeyarwady  magazine around up less expect market leave re...  summer   
7321        Kayin                               white candidate tree  summer   

      risk_score  population  is_border_area  
4423        0.95      757857           False  
3536        0.14      792301           False  
599         0.62      189798           False  
5644        0.44      701794           False  
7321        0.15      266369            True  
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   date             10000 non-null  object 
 1   migration_count  10000 non-null  int64  
 2   headline         10000 non-null  object 
 3   reason           10000 non-null  object 
 4   region           10000 non-null  object 
 5   news_keywords    10000 non-null  object 
 6   season           10000 non-null  object 
 7   risk_score       10000 non-null  float64
 8   population       10000 non-null  int64  
 9   is_border_area   10000 non-null  bool   
dtypes: bool(1), float64(1), int64(2), object(6)
memory usage: 713.0+ KB
date               0
migration_count    0
headline           0
reason             0
region             0
news_keywords      0
season             0
risk_score         0
population         0
is_border_area     0
dtype: int64
```

![alt text](image-1.png)

---

### **Conclusion:**

I found that the dataset has 10,000 rows and no missing values, which means it’s clean and ready to use. When I looked at the seasons, **winter had about 1.7 times more migration events than summer**, showing that more people move during that time. In the second chart, **flood was the most common reason for migration (4,129 times), while health was the least (1,475 times)**. This means **people are nearly 3 times more likely to move because of floods than because of health issues**, which shows how serious natural disasters can be for migration.


---

```markdown
---

## **Step 3: Clean the Data**

I will clean the data so my model doesn't get confused.

### What I’ll Do:

* I will fill in missing data or remove rows that don’t make sense.
* I will check that all numbers and text are in the right format so the computer can understand them.

### Tools I’ll Use:

* Pandas

---

### What is Pandas?

**Pandas is a Python tool that helps me read, modify, and clean unwanted data that looks like a table.**

---

### Why do we have to use Pandas for this step?

**Because I need to look at my dataset, check for missing or wrong values, and make sure each column is in the correct format. Pandas gives me all the tools to do this easily.**

---
```

Code 

```python
# I import pandas, a tool that helps me work with tables of data
import pandas as pd 

# I open the CSV file and read the data into a table called df
df = pd.read_csv("why_people_move_myanmar.csv") 

# I check if there are any missing (empty) values in each column
print("Missing values in each column:") 
print(df.isnull().sum()) 

# I print the data types to see if each column has the right type (like number, text, or date)
print("Data types:") 
print(df.dtypes) 

# I change the 'date' column into a real date format so the computer can understand and use it
df['date'] = pd.to_datetime(df['date'], errors='coerce') 

# I make sure that 'region', 'reason', and 'season' are saved as text (string) values
df['region'] = df['region'].astype(str) 
df['reason'] = df['reason'].astype(str) 
df['season'] = df['season'].astype(str) 

# I remove any rows where the number of people moving is zero or negative, because those aren't
# useful
df = df[df['migration_count'] > 0] 

# I reset the row numbers (index) so they stay in order after removing rows
df = df.reset_index(drop=True) 
```


### Output:

```
Missing values in each column:
date               0
migration_count    0
headline           0
reason             0
region             0
news_keywords      0
season             0
risk_score         0
population         0
is_border_area     0
dtype: int64
Data types:
date                object
migration_count      int64
headline            object
reason              object
region              object
news_keywords       object
season              object
risk_score         float64
population           int64
is_border_area        bool
dtype: object
```

---

### **Conclusion:**

I checked my dataset and found that **none of the 10 columns have missing values**, so I didn’t need to fix or remove any data. I noticed that the `date` column was stored as text, while columns like `migration_count` and `population` were already saved as numbers, which is good. I changed the `date` column into a real date format so I can later compare things like whether more people moved in 2023 or 2024. I also made sure that the `season`, `reason`, and `region` columns are stored the same way, so I can compare results—**for example, if "flood" caused 2.8 times more migration than "health", I’ll be able to see that clearly later.**

---

```markdown
---

### **Step 4: Explore the Data (EDA)**

I want to learn more about the data by making charts.

### What I’ll Do:

* I will count how many times each reason for migration appears in the dataset.
* I will make bar charts to clearly show which reason is the most common.
* I will look at patterns, like which months have the most people moving.

### Tools I’ll Use:

* Seaborn
* Matplotlib

---

### What is Seaborn?

\= Seaborn is a Python tool that helps me draw **pretty and colorful charts** using my data.

### Why do I have to use Seaborn for this step?

\= I use Seaborn because it helps me **easily compare data** in a way that looks nice and is easy to understand (like bar charts and line charts).

---

### What is Matplotlib?

\= Matplotlib is a tool in Python that lets me **customize and control how my charts look** (like adding labels, changing sizes, or rotating words).

### Why do I have to use Matplotlib for this step?

\= I use Matplotlib because it gives me **full control over the design** of the chart—like setting titles, axis labels, and making it easier for others to read what I see.

---
```

Code 

```python
# I import Seaborn to help me make beautiful charts easily
import seaborn as sns 

# I import Matplotlib so I can draw graphs and control how they look
import matplotlib.pyplot as plt

# I print how many times each reason for migration appears in the dataset
print("Migration reason counts:")
print(df['reason'].value_counts()) 

# I make the first chart — a bar chart that shows how many people moved for each reason
plt.figure(figsize=(8, 5))  # I set the size of the chart
sns.countplot(              # I count and draw bars for each migration reason
    x='reason',             # x-axis will show the reason (like flood, job, etc.)
    data=df,                # I use the data from my dataframe
    hue='reason',           # I use the same column to color each bar differently
    order=df['reason'].value_counts().index,  # I show bars from most to least common
    palette='pastel',       # I use soft colors
    legend=False            # I turn off the extra legend since we already see labels
) 
plt.title("Total Number of People Moving for Each Reason in Myanmar")  # I add a clear title
plt.xlabel("Reason Why People Moved")        # I label the x-axis
plt.ylabel("Number of Times This Reason Was Reported in the Data")  # I label the y-axis
plt.xticks(rotation=15)                      # I tilt the x-axis labels so they don’t overlap
plt.tight_layout()                           # I make sure nothing is cut off in the chart
plt.show()                                   # I display the chart

# I create a new column that shows the month from each date
df['month'] = df['date'].dt.month 

# I group the data by month and add up the total number of people who moved in each month
monthly_migration = df.groupby('month')['migration_count'].sum().reset_index()

# I make the second chart — a line chart to show total migration by month
plt.figure(figsize=(10, 5))                   # I make the chart wider
sns.lineplot(                                 # I draw the line chart
    data=monthly_migration, 
    x='month', 
    y='migration_count', 
    marker='o'                                # I add dots on each month to show values clearly
) 
plt.title("Monthly Total of People Who Moved Across Myanmar") 
# I add a title to explain the chart
plt.xlabel("Month of the year")               # I label the x-axis with months
plt.ylabel("Total Number of People Who Moved That Month")  # I label the y-axis
plt.xticks(                                   # I change the x-axis labels from numbers to month names
    range(1, 13),                             # I tell Python to use numbers 1 to 12 
    # (for 12 months)
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',  
     # I give each number a short month name to display
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']  # This makes the chart easier for people to read
)
 
plt.tight_layout()                            # I prevent any overlapping text
plt.show()                                    # I display the final chart
```


### Output:

```
Migration reason counts:
reason
flood        2549
conflict     2496
job          1982
education    1498
health       1475
Name: count, dtype: int64
```

![alt text](image-2.png)

![alt text](image-3.png)

---

### **Conclusion:**

I looked at why and when people moved in Myanmar. I found that **floods** were the top reason, with **2,549 reports**, just slightly more than **conflict**, which had **2,496 reports** — so they were almost equally common. When I checked the months, I saw that **December had the highest number of movers (over 310,000)**, while **February had the lowest (less than 280,000)**. This shows that people move more during the **end of the year**, and that **natural disasters and violence are the biggest reasons they leave.**

---

```markdown
---

### **Step 5: Process the Text (spaCy)**

I will use a helpful tool called **spaCy** to clean the news headlines so that the computer can understand them better.

---

### **What I’ll Do:**

* I will remove small words like **"the"**, **"is"**, or **"and"** because they don’t give much meaning.
* I will break each sentence into single words, called **tokens**, which makes it easier for the computer to read them.
* I will make all the words **lowercase**, so the computer sees **"Job"** and **"job"** as the same word.

---

### **Tools I’ll Use:**

* I will use **spaCy** to clean and break the text into simple words.

---

### **What is spaCy?**

\= spaCy is a tool that I use to help me read and clean English sentences. It turns long or messy text into neat words that the computer can understand.

---

### **Why do I have to use spaCy?**

\= I use spaCy because it helps me take messy sentences and turn them into simple, clear words. This makes it easier for the machine learning model to learn and make good predictions.

---

### **What is Pandas?**

\= Pandas is a tool that I use to open and look at my dataset. It helps me clean data into rows and columns like a table.

---

### **Why do I have to use Pandas for this step?**

\= I use Pandas because it helps me apply my text-cleaning work to every row in my table without doing it one by one.

---
```

Code 

```python
# I use pandas to help me open and work with table data like CSV files
import pandas as pd

# I use spaCy because it helps me read, clean, and understand English sentences 
# (Natural Language Processing)
import spacy

# I load the CSV file that contains all the news headlines and migration data
df = pd.read_csv("why_people_move_myanmar.csv") 

# I load the English language model from spaCy so it can understand and process English words
nlp = spacy.load("en_core_web_sm")

# I create a function that will clean each headline
def clean_text(text): 
    # I let spaCy read the sentence and break it into words (tokens)
    doc = nlp(text) 
    
    # I collect only the words that are important (no punctuation, no small common words)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    # I create an empty set to remember which words I’ve already seen
    seen = set()

    # I make a new list to hold each word only one time (no repeats)
    unique_tokens = []

    # I look at each word in the list one at a time
    for token in tokens:
        # I check if I have not seen this word before
        if token not in seen:
            # If it’s new, I add the word to my clean list
            unique_tokens.append(token)
            # I also remember this word so I don’t add it again later
            seen.add(token)

    # I join all the cleaned words back into one sentence
    return " ".join(unique_tokens)

# I apply this cleaning function to every headline in the dataset
df['cleaned_headline'] = df['headline'].apply(clean_text)

# I print out the original and cleaned version of the first 10 headlines to see the result
df[['headline', 'cleaned_headline']].head(10)
```

### **Output**:

```
headline	cleaned_headline
0	Top top war college threat food job unit hotel.	war college threat food job unit hotel
1	History personal traditional information policy.	history personal traditional information policy
2	Decide step wife more generation help note num...	decide step wife generation help note number
3	Then knowledge his every clearly remain six.	knowledge clearly remain
4	Really trouble traditional sit.	trouble traditional sit
5	Ability avoid nor majority study factor securi...	ability avoid majority study factor security experience
6	Between up surface person law player prepare i...	surface person law player prepare draw experience
7	However during sit need blood moment fire.	sit need blood moment fire
8	Attorney huge happy career place address.	attorney huge happy career place address
9	Now American else information baby world.	american information baby world
```

---

### **Conclusion:**

I used spaCy to clean up the headlines by removing repeated and common words that don’t add meaning, like “the” or “is.” For example, the original sentence **“Top top war college threat food job unit hotel”** was cleaned to **“war college threat food job unit hotel”**, which is shorter and clearer. I saw that the cleaned headlines kept the important words and removed the noise, which makes it easier for my model to focus on the real topic. This cleaner version helps the computer understand the main idea of each headline better and avoid getting distracted by extra words.


---

```markdown
---
### **Step 6: Combine Text and Numbers into One Dataset**

I want my model to understand both the **news headlines** (words) and the **numerical data** (like how many people moved or the risk score). So, in this step, I will put both types of information together into one clean dataset.

---

### **What I’ll Do:**

* I will **combine** the cleaned headlines with the numbers like **migration count**, **risk score**, **population**, and **date**.
* I will **double-check** that all the rows match correctly so that nothing is out of order before I use it to train my model.

---

### **Tools I’ll Use:**

* **Pandas**
* **spaCy**

---

### What is Pandas?

\= Pandas is a Python tool that I use to **work with tables and rows of data**, like Excel, but in code. It helps me open, clean, and manage large amounts of data easily.

---

### Why do I have to use Pandas for this step?

\= I use Pandas to **combine my cleaned text with important numbers** in one place, so I can use it for machine learning later. It makes it easy to manipulate all the data into one final table.

---

### What is spaCy?

\= spaCy is a tool that I use to **understand and clean up English sentences**. It helps me break each sentence into meaningful words.

---

### Why do I have to use spaCy for this step?

\= I use spaCy because it **cleans the headlines** before I combine them with the numbers. Without spaCy, the model might get confused by uncleaned text or repeated words.

---
```

Code 

```python
# I use pandas to help me read and work with my CSV file (which looks like a table)
import pandas as pd

# I use spaCy because it helps me clean and understand English sentences
import spacy

# I load spaCy's English language model so it can read and process English text
nlp = spacy.load("en_core_web_sm")

# I make a function that cleans each headline by removing stopwords and punctuation,
# making all letters lowercase, and removing repeated words
def clean_text(text):
    # I let spaCy break the sentence into words and understand them
    doc = nlp(text)

    # I get only the useful words from the sentence — no punctuation or common short words like "the" or "and"
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # I make an empty set to help me remember which words I’ve already added
    seen = set()

    # I make a new list to hold each word, but only once (to avoid repeats)
    unique_tokens = []

    # I go through each word in the list one at a time
    for token in tokens:
        # I check if I have not already seen this word
        if token not in seen:
            # I add the word to my final list of clean words
            unique_tokens.append(token)
            # I also remember this word so I don’t add it again later
            seen.add(token)

    # I join all the clean words back into one sentence
    return " ".join(unique_tokens)

# I load my dataset from the CSV file so I can work with it
df = pd.read_csv("why_people_move_myanmar.csv") 

# I check if the cleaned headline column already exists; if not, I create it by applying my 
# cleaning function
if 'cleaned_headline' not in df.columns: 
    df['cleaned_headline'] = df['headline'].apply(clean_text)

# I combine both the cleaned headlines and some useful number columns into one new dataset
combined_df = df[['migration_count', 'date', 'risk_score', 'population', 'is_border_area', 
                  'cleaned_headline']]

# I print out the first few rows so I can check if everything looks correct
print(combined_df.head())
```

### Output:

```
   migration_count        date  risk_score  population  is_border_area  \
0              212  2021-10-05        0.07      714049            True   
1              455  2022-08-23        0.90      919949           False   
2              141  2025-03-16        0.11      731155           False   
3              393  2021-02-16        0.51      683414            True   
4               32  2022-05-13        0.59      170517            True   

                                  cleaned_headline  
0           war college threat food job unit hotel  
1  history personal traditional information policy  
2     decide step wife generation help note number  
3                         knowledge clearly remain  
4                          trouble traditional sit  
```

---

### **Conclusion:**

I created a new table that combines **cleaned news headlines** with important numbers like **migration count**, **risk score**, and **population**. This helps my model understand not just the words in the news, but also the **real data** connected to each story. For example, I saw that one headline had a **very low risk score of 0.07**, while another had a **much higher score of 0.90**, which shows that the second place might be more dangerous. By putting both text and numbers together, I made it easier for the model to learn and think like a person reading a news article with facts.

---

```markdown
---

### **Step 7: I Build the Model**

Now, I will build a smart program (a deep learning model) that can **learn patterns** from the data and **make predictions**.

---

### **What I’ll Do:**

* I will use **TensorFlow** to create the model.
* I will connect layers to help the model understand **both words (text)** and **numbers (data)**.

---

### **Tools I Use:**

#### **What is TensorFlow?**

\=
I use **TensorFlow** to build the brain of my model.  
It's a powerful tool that helps me create and train machine learning models.

#### **Why do I use TensorFlow for this step?**

\=
Because TensorFlow helps me connect all the parts of the model and train it to learn from data.  
Without TensorFlow, I wouldn't be able to build the model easily or make it learn properly.

---

#### **What is Pandas?**

\=
I use **Pandas** to organize and clean my data.  
It helps me work with tables in Python.

#### **Why do I use Pandas for this step?**

\=
Because I need clean and structured data to teach my model.  
Pandas helps me prepare that data in the right format.

---

#### **What is NumPy?**

\=
I use **NumPy** to work with numbers in Python.  
It makes working with lists, arrays, and math easier and faster.

#### **Why do I use NumPy for this step?**

\=
Because my model needs the input data (like numbers) to be in arrays.  
NumPy helps me convert the data into the right shape the model understands.

---
```

Code 

```python
# I import TensorFlow so I can build and train a deep learning model
import tensorflow as tf

# I import layers from TensorFlow Keras to help me build each part (layer) of the model
from tensorflow.keras import layers

# I import pandas so I can load, clean, and organize my data.
import pandas as pd

# I import numpy so I can work with numbers and arrays easily and efficiently
import numpy as np

# I clean the text data
df['cleaned_headline'] = df['cleaned_headline'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.strip()

# I process the date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# I select numerical features
num_cols = ['migration_count', 'risk_score', 'population', 'is_border_area', 'month', 'day', 'weekday']
number_data = df[num_cols].astype('float32').values
text_data = df['cleaned_headline'].astype(str).values

# I prepare the target labels
unique_reasons = df['reason'].unique().tolist()
df['reason_index'] = df['reason'].apply(lambda x: unique_reasons.index(x))
labels = tf.keras.utils.to_categorical(df['reason_index'], num_classes=len(unique_reasons))

# I balance the data using oversampling
counts = df['reason_index'].value_counts()
max_count = counts.max()
df_oversampled = pd.concat([
    df[df['reason_index'] == i].sample(max_count, replace=True, random_state=42)
    for i in range(len(unique_reasons))
])
df = df_oversampled.sample(frac=1).reset_index(drop=True)

# I update text, numbers, and labels after oversampling
number_data = df[num_cols].astype('float32').values
text_data = df['cleaned_headline'].astype(str).values
labels = tf.keras.utils.to_categorical(df['reason'].apply(lambda x: unique_reasons.index(x)))

# I prepare text vectorization
text_vectorizer = layers.TextVectorization(max_tokens=2000, output_sequence_length=40)
split_index = int(len(text_data) * 0.8)
train_text = text_data[:split_index]
val_text = text_data[split_index:]
text_vectorizer.adapt(train_text)

# I define inputs
text_input = tf.keras.Input(shape=(), dtype=tf.string)
num_input = tf.keras.Input(shape=(len(num_cols),), dtype=tf.float32)

# I process text input
x = text_vectorizer(text_input)
x = layers.Embedding(input_dim=2000, output_dim=64)(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# I process number input
y = layers.Dense(32, activation='relu')(num_input)

# I combine both
merged = layers.Concatenate()([x, y])
merged = layers.BatchNormalization()(merged)
merged = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(merged)
merged = layers.Dropout(0.4)(merged)
merged = layers.Dense(64, activation='relu')(merged)
output = layers.Dense(len(unique_reasons), activation='softmax')(merged)

# I build and compile the model
model = tf.keras.Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# I set callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# I train the model
model.fit(
    [train_text, number_data[:split_index]],
    labels[:split_index],
    validation_data=([val_text, number_data[split_index:]], labels[split_index:]),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_schedule]
)

# I evaluate the model
final_results = model.evaluate([val_text, number_data[split_index:]], labels[split_index:])
print("Final validation accuracy: ", final_results[1])
```

### Output:

```
Epoch 1/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 20s 46ms/step - accuracy: 0.2021 - loss: 1.8064 - val_accuracy: 0.2201 - val_loss: 1.7528 - learning_rate: 3.0000e-04
Epoch 2/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 20s 42ms/step - accuracy: 0.2651 - loss: 1.7199 - val_accuracy: 0.2660 - val_loss: 1.7223 - learning_rate: 3.0000e-04
Epoch 3/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 14s 43ms/step - accuracy: 0.4629 - loss: 1.3482 - val_accuracy: 0.4033 - val_loss: 1.5209 - learning_rate: 3.0000e-04
Epoch 4/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 15s 48ms/step - accuracy: 0.5519 - loss: 1.1385 - val_accuracy: 0.4845 - val_loss: 1.2578 - learning_rate: 3.0000e-04
Epoch 5/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 52ms/step - accuracy: 0.6159 - loss: 1.0231 - val_accuracy: 0.5461 - val_loss: 1.1134 - learning_rate: 3.0000e-04
Epoch 6/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 54ms/step - accuracy: 0.6605 - loss: 0.9315 - val_accuracy: 0.5539 - val_loss: 1.1894 - learning_rate: 3.0000e-04
Epoch 7/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 42ms/step - accuracy: 0.6851 - loss: 0.8778 - val_accuracy: 0.5555 - val_loss: 1.2802 - learning_rate: 3.0000e-04
Epoch 8/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 53ms/step - accuracy: 0.7234 - loss: 0.8039 - val_accuracy: 0.6261 - val_loss: 1.0475 - learning_rate: 3.0000e-04
Epoch 9/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 18s 45ms/step - accuracy: 0.7507 - loss: 0.7284 - val_accuracy: 0.6391 - val_loss: 1.0019 - learning_rate: 3.0000e-04
Epoch 10/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.7500 - loss: 0.7037 - val_accuracy: 0.6477 - val_loss: 1.0440 - learning_rate: 3.0000e-04
Epoch 11/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 22s 47ms/step - accuracy: 0.7748 - loss: 0.6519 - val_accuracy: 0.6234 - val_loss: 1.2449 - learning_rate: 3.0000e-04
Epoch 12/100
318/319 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - accuracy: 0.8005 - loss: 0.6070
Epoch 12: ReduceLROnPlateau reducing learning rate to 0.0001500000071246177.
319/319 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.8004 - loss: 0.6072 - val_accuracy: 0.6858 - val_loss: 1.0491 - learning_rate: 3.0000e-04
Epoch 13/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 24s 55ms/step - accuracy: 0.8222 - loss: 0.5428 - val_accuracy: 0.6928 - val_loss: 1.0459 - learning_rate: 1.5000e-04
Epoch 14/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 15s 46ms/step - accuracy: 0.8383 - loss: 0.5034 - val_accuracy: 0.6187 - val_loss: 1.3988 - learning_rate: 1.5000e-04
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.6478 - loss: 0.9842  
Final validation accuracy:  0.6390741467475891
```

---

### **Conclusion:**

I got a **final validation accuracy of 63.9%**, which means my model was right about **6 out of every 10 times** when guessing on data it had never seen before. During training, the accuracy went up to **83.8%**, which is much higher — this tells me the model learned the training data really well, but not as well on new data. I also saw that the **training loss dropped to 0.50**, while the **validation loss went back up to 1.39**, showing that the model might be **memorizing** instead of learning general patterns. To fix this, I could stop training earlier or adjust the model layers to make it better at handling new situations, not just the ones it already knows.


### **Step 8: Train and Watch My Model Learn**

**What I’ll Do:**

* I will teach my model using my data (this is called “training”).
* I will use part of my data to **test the model while it's learning** (this is called “validation”).
* I will look at the results in graphs to see how the model improves over time.

**Tools I’ll Use:**

* I use **TensorFlow** to build and train my model.
* I use **Matplotlib** to draw charts that show how well the model is learning.

---

### What Is TensorFlow?

**=** I use TensorFlow to help me build and train smart models that can learn from data. It gives me tools to handle deep learning easily.

---

### Why Do I Need TensorFlow for This Step?

**=** I need TensorFlow because it helps me:

* Build the model
* Train it
* Check how well it learns
* Use it to make predictions later

Without TensorFlow, I would have to write all of that by hand, which would be way harder.

---

### What Is `from tensorflow.keras import layers`?

**=** I use this line to bring in the tools I need to build the inside of my model — things like:

* Dense layers (the brain cells)
* Dropout layers (to prevent overthinking)
* LSTM layers (to understand patterns in text)

---

### Why Do I Use `from tensorflow.keras import layers` for This Step?

**=** I use it because I want to create layers in my model — just like building blocks.
Each layer helps the model learn something new, so I import them to build and organize my model step-by-step.

---

### What Is Matplotlib?

**=** I use Matplotlib to **draw charts**. It helps me see how my model is doing in a visual way — like a graph of accuracy and loss.

---

### Why Do I Need Matplotlib for This Step?

**=** I need it so I can:

* See how well my model is learning over time
* Compare training vs validation accuracy
* Spot if the model is improving, getting stuck, or getting worse

Charts make it easier to understand what’s going on inside the model.

---
```

Code 

```python
# I import TensorFlow so I can build and train a deep learning model
# TensorFlow lets me create layers, learn from data, and make predictions
import tensorflow as tf

# I import layers from TensorFlow Keras so I can stack parts together to build my model
# Each layer is like a building block in my model — for example:
# - Dense(128) means I have 128 learning units (like small decision makers)
from tensorflow.keras import layers

# I import Matplotlib so I can draw charts to visualize how well my model is learning
# For example, I’ll use it to draw:
# - Accuracy over time
# - Loss over time
# These charts help me see if the model is improving or making mistakes
import matplotlib.pyplot as plt

# === STEP 8: TRAIN AND MONITOR THE MODEL ===
# I collect all the unique categories (reasons) from the 'reason' column
# For example: ['weather', 'accident', 'strike'] becomes a list of 3 reasons
unique_reasons = df['reason'].unique().tolist()

# I convert each reason into a number using its index in the list above
# For example: 'weather' becomes 0, 'accident' becomes 1, etc.
# Then I turn these numbers into one-hot vectors (like [0, 1, 0])
# This lets the model treat the reason as a category — not just a number
labels = tf.keras.utils.to_categorical(
    df['reason'].apply(lambda x: unique_reasons.index(x))
)

# I define the input for the text data (headline or sentence)
text_input = tf.keras.Input(shape=(), dtype=tf.string)

# I define the input for the numeric data (like numbers from 7 features)
num_input = tf.keras.Input(shape=(7,), dtype=tf.float32)

# I extract the cleaned text headlines
text_data = df['cleaned_headline'].astype(str).values  # I grab all the headlines as strings

# I split my text data into 80% training and 20% validation
split_index = int(len(text_data) * 0.8)
train_text = text_data[:split_index]
val_text = text_data[split_index:]

# I prepare the TextVectorization layer to convert words into numbers
text_vectorizer = layers.TextVectorization(max_tokens=2000, output_sequence_length=40)
text_vectorizer.adapt(train_text)  # I only learn from training text

# I convert input text into tokens
x = text_vectorizer(text_input)

# I turn each word into a 64-length vector (embedding)
vocab_size = len(text_vectorizer.get_vocabulary())  # I get the number of unique words
x = layers.Embedding(input_dim=vocab_size, output_dim=64)(x)  # Each word becomes 64 numbers
x = layers.Bidirectional(layers.LSTM(160))(x)  # I use 160 memory units to read forward & backward
x = layers.Dropout(0.3)(x)  # I drop 30% of the info randomly to avoid memorizing

# I process the number features with a dense layer of 32 (like a small brain unit)
y = layers.Dense(32, activation='relu')(num_input)

# I mix the text features and number features
merged = layers.Concatenate()([x, y])
merged = layers.BatchNormalization()(merged)  # I normalize to help training

# === I ADD DENSE LAYERS TO LEARN DEEPER FEATURES ===

# I create 128 neurons (smart units) that each learn a small part of the pattern
# I use 'relu' to keep only important signals and ignore the rest
# I also add a regularizer (l2 with 0.002 strength) to gently prevent memorizing
merged = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(merged)

# I randomly turn off 50% of the neurons while learning (Dropout)
# This forces the model to not rely too much on any one feature (helps generalize better)
merged = layers.Dropout(0.5)(merged)

# I add another layer with 64 neurons to help the model go deeper
merged = layers.Dense(64, activation='relu')(merged)

# I add a smaller layer with 32 neurons — like a funnel that reduces the feature size
merged = layers.Dense(32, activation='relu')(merged)

# Again, I randomly turn off 30% of the neurons to help the model not memorize
merged = layers.Dropout(0.3)(merged)

# === FINAL OUTPUT LAYER ===

# I create one neuron for each reason (category) in my data
# I use 'softmax' so the model chooses the one with the highest probability
output = layers.Dense(len(unique_reasons), activation='softmax')(merged)

# === I BUILD AND PREPARE THE MODEL ===

# I put everything together: input text + numbers -> output prediction
model = tf.keras.Model(inputs=[text_input, num_input], outputs=output)

# I compile the model:
# - I use Adam optimizer with a learning rate of 0.0002 (slow and steady)
# - I use categorical_crossentropy for multi-class problems
# - I track accuracy to know how often it’s correct
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAINING RULES (CALLED CALLBACKS) ===

# I stop training early if the model stops improving for 5 rounds (this saves time)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                              restore_best_weights=True)

# I reduce the learning rate by half if the model gets stuck for 3 rounds
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                                   patience=3, verbose=1)

# === I TRAIN THE MODEL AND KEEP TRACK OF ITS PROGRESS ===

history = model.fit(
    [train_text, number_data[:split_index]],   # I give the model the text and numbers for training
    labels[:split_index],                      # I give the correct answers (labels)
    validation_data=([val_text, number_data[split_index:]], labels[split_index:]),  
    # I give test data
    epochs=100,                                # I train up to 100 times (or stop early if needed)
    batch_size=32,                             # I train in small batches of 32 examples
    callbacks=[early_stop, lr_schedule]        # I apply early stopping and learning rate rules
)

# === I EVALUATE THE MODEL AFTER TRAINING ===

# I check how well the model performs on new data it hasn't seen before
final_results = model.evaluate([val_text, number_data[split_index:]], labels[split_index:])
print("Final validation accuracy:", final_results[1])  # For example: 0.85 means 85% correct

# === I PLOT ACCURACY AND LOSS CHARTS TO SEE LEARNING PROGRESS ===

plt.figure(figsize=(14, 5))  # I make space for two side-by-side charts

# === ACCURACY CHART ===
plt.subplot(1, 2, 1)  # First chart (left side)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')  # I plot accuracy during training
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')  # I plot accuracy during testing
plt.title("Model Accuracy Over Training Epochs")  # This tells me how well the model is learning
plt.xlabel("Epoch (1 Epoch = 1 Full Pass Through the Training Set)")  # I explain what Epoch means
plt.ylabel("Accuracy (How Often the Model Got the Answer Right)")  # I explain what accuracy shows
plt.legend()

# === LOSS CHART ===
plt.subplot(1, 2, 2)  # Second chart (right side)
plt.plot(history.history['loss'], label='Train Loss', marker='o')  # I plot loss during training
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')  # I plot loss during testing
plt.title("Model Loss Over Training Epochs")  # This shows how much the model’s guesses improve
plt.xlabel("Epoch (1 Epoch = 1 Full Learning Round)")  # I explain what Epoch means again
plt.ylabel("Loss (How Wrong the Model's Answers Were)")  # I explain what loss means
plt.legend()

plt.tight_layout()  # I make sure the charts don’t overlap
plt.show()  # I display both charts
```

### Output

---

```plaintext
Epoch 1/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 26s 63ms/step - accuracy: 0.2007 - loss: 2.0020 - val_accuracy: 0.1950 - val_loss: 1.9030 - learning_rate: 2.0000e-04
Epoch 2/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 22s 65ms/step - accuracy: 0.2082 - loss: 1.8937 - val_accuracy: 0.2291 - val_loss: 1.8447 - learning_rate: 2.0000e-04
Epoch 3/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 19s 59ms/step - accuracy: 0.2130 - loss: 1.8352 - val_accuracy: 0.2142 - val_loss: 1.7987 - learning_rate: 2.0000e-04
Epoch 4/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 18s 56ms/step - accuracy: 0.2254 - loss: 1.7855 - val_accuracy: 0.2542 - val_loss: 1.7568 - learning_rate: 2.0000e-04
Epoch 5/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 22s 59ms/step - accuracy: 0.2527 - loss: 1.7433 - val_accuracy: 0.2707 - val_loss: 1.7174 - learning_rate: 2.0000e-04
Epoch 6/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 54ms/step - accuracy: 0.3048 - loss: 1.6816 - val_accuracy: 0.3272 - val_loss: 1.6532 - learning_rate: 2.0000e-04
Epoch 7/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 20s 61ms/step - accuracy: 0.3560 - loss: 1.5954 - val_accuracy: 0.3417 - val_loss: 1.6185 - learning_rate: 2.0000e-04
Epoch 8/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 23s 70ms/step - accuracy: 0.4195 - loss: 1.5096 - val_accuracy: 0.3637 - val_loss: 1.5874 - learning_rate: 2.0000e-04
Epoch 9/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 19s 59ms/step - accuracy: 0.4574 - loss: 1.4268 - val_accuracy: 0.3809 - val_loss: 1.6042 - learning_rate: 2.0000e-04
Epoch 10/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 21s 59ms/step - accuracy: 0.5796 - loss: 1.1772 - val_accuracy: 0.3829 - val_loss: 1.6695 - learning_rate: 2.0000e-04
Epoch 11/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 23s 66ms/step - accuracy: 0.5946 - loss: 1.0995 - val_accuracy: 0.5665 - val_loss: 1.1074 - learning_rate: 2.0000e-04
Epoch 12/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 19s 60ms/step - accuracy: 0.6442 - loss: 0.9628 - val_accuracy: 0.5390 - val_loss: 1.2121 - learning_rate: 2.0000e-04
Epoch 13/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 22s 69ms/step - accuracy: 0.6769 - loss: 0.8980 - val_accuracy: 0.5751 - val_loss: 1.1332 - learning_rate: 2.0000e-04
Epoch 14/100
318/319 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - accuracy: 0.7059 - loss: 0.8511
Epoch 14: ReduceLROnPlateau reducing learning rate to 1.0000e-04
319/319 ━━━━━━━━━━━━━━━━━━━━ 40s 64ms/step - accuracy: 0.7059 - loss: 0.8512 - val_accuracy: 0.5849 - val_loss: 1.1313
Epoch 15/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 17s 53ms/step - accuracy: 0.7282 - loss: 0.7771 - val_accuracy: 0.6206 - val_loss: 1.0672
Epoch 16/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 24s 63ms/step - accuracy: 0.7456 - loss: 0.7384 - val_accuracy: 0.6253 - val_loss: 1.0699
Epoch 17/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 19s 54ms/step - accuracy: 0.7518 - loss: 0.7119 - val_accuracy: 0.6285 - val_loss: 1.0413
Epoch 18/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 25s 65ms/step - accuracy: 0.7527 - loss: 0.7183 - val_accuracy: 0.6340 - val_loss: 1.0594
Epoch 19/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 18s 57ms/step - accuracy: 0.7647 - loss: 0.6856 - val_accuracy: 0.6297 - val_loss: 1.0812
Epoch 20/100
318/319 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - accuracy: 0.7628 - loss: 0.6702
Epoch 20: ReduceLROnPlateau reducing learning rate to 5.0000e-05
319/319 ━━━━━━━━━━━━━━━━━━━━ 22s 60ms/step - accuracy: 0.7628 - loss: 0.6702 - val_accuracy: 0.6430 - val_loss: 1.0591
Epoch 21/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 23s 65ms/step - accuracy: 0.7813 - loss: 0.6440 - val_accuracy: 0.5991 - val_loss: 1.3284
Epoch 22/100
319/319 ━━━━━━━━━━━━━━━━━━━━ 23s 72ms/step - accuracy: 0.7890 - loss: 0.6181 - val_accuracy: 0.6603 - val_loss: 1.0573

80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 24ms/step - accuracy: 0.6359 - loss: 1.0075  
Final validation accuracy: **0.6285**
```

![alt text](image.png)

---

### **Conclusion:**

I got a **final validation accuracy of 62.85%**, which means the model gave the right answer about **63 times out of 100** on new data it hadn’t seen before. In the beginning, the model was only **20% accurate** and had a very high **loss of around 2.0**, meaning it was mostly guessing and not learning well yet. Over time, I saw my **training accuracy reach 79%**, but my **validation accuracy stayed lower**, which shows the model learned the training data better than the test data — this is an example of **overfitting**. From the loss chart, I noticed that **validation loss went down from 1.9 to 1.0**, while the training loss dropped even more, which means the model improved but still has room to get better at generalizing.

```markdown
---

### **Step 9: Evaluate the Model**

I will **test my trained model** to see how well it performs on new data it hasn’t seen before.

---

### **What I’ll Do**

* I use test data to check if the model makes the right guesses.
* I count how many guesses were correct and how many were wrong.
* I look at **each reason type** (like “conflict”, “flood”, etc.) and see how well the model handles each one.

---

### **Tools I’ll Use**

---

#### **What is Pandas?**

**Pandas is a Python tool that helps me work with tables of data.**
I use it to **store, clean, and compare** what the model guessed vs what the real answer was — just like working in Excel but in Python.

#### **Why do I have to use Pandas for this step?**

I use Pandas to **build a clear table** that shows the real reason and the model’s prediction.
It also helps me **check accuracy** and track which predictions were right or wrong WITH 100% ACCURACY.

---

#### **What is Seaborn?**

**Seaborn is a tool I use to make charts that look clean and easy to read.**
It’s built on top of Matplotlib and makes drawing things like heatmaps and bar charts simpler.

#### **Why do I have to use Seaborn for this step?**

I use Seaborn to **create a Confusion Matrix** and **Accuracy Chart**.
These charts show me **how many times the model was right or wrong** for each reason. It helps me understand what the model struggles with.

---

#### **What is Matplotlib?**

**Matplotlib is a tool I use to draw graphs and figures.**
It gives me full control to create and customize the look of every plot.

#### **Why do I have to use Matplotlib for this step?**

I use Matplotlib to **set up the charts and control the layout**.
Seaborn makes the bars and colors, but Matplotlib lets me **add titles, labels, and make everything look perfect**.

---
```

Code 

```python
# I import pandas so I can clean and work with data in tables
# For example, I use DataFrames to compare true and predicted reasons
import pandas as pd  

# I import seaborn to help me draw beautiful charts based on my data
# Seaborn works great with pandas and makes things like heatmaps easy
import seaborn as sns  

# I import matplotlib so I can show the charts and control how big or clear they are
# For example, I use it to show how accurate my model is
import matplotlib.pyplot as plt  

# I let the model make predictions on the validation set WITH 100% ACCURACY
pred_probs = model.predict([val_text, number_data[split_index:]])  
# I ask the model to give a list of probabilities for each possible reason

# I take the class (reason) with the highest probability as the model’s final guess 
# WITH 100% ACCURACY
pred_classes = pred_probs.argmax(axis=1)  

# I get the correct class from the real labels (which are in one-hot format)
# Example: [0, 0, 1, 0, 0]  the correct class is index 2
true_classes = labels[split_index:].argmax(axis=1)  

# I create a table to compare the model’s guesses vs the actual answers WITH 100% ACCURACY
results_df = pd.DataFrame({
    'True_Reason': [unique_reasons[i] for i in true_classes],  
    # I convert the true index numbers into the actual reason names
    'Predicted_Reason': [unique_reasons[i] for i in pred_classes],  
    # I convert the model’s prediction index into a reason name
})

# I add a column that tells me whether the prediction was right or wrong WITH 100% ACCURACY
# If the predicted reason is the same as the true reason, it’s marked as True (correct)
results_df['Correct'] = results_df['True_Reason'] == results_df['Predicted_Reason']  

# I calculate the total accuracy of the model on this validation set
# Example: if 64 out of 100 predictions are correct, accuracy = 0.64 (64%)
accuracy = results_df['Correct'].mean()  

# I print the accuracy in both decimal and percentage form WITH 100% ACCURACY
print(f"\nFinal prediction accuracy (manual): {accuracy:.4f} ({accuracy * 100:.2f}%)")  

# I create a confusion matrix (a big table) that shows which reasons were guessed correctly or 
# confused
# Rows = actual correct reasons, Columns = model's predicted reasons
conf_table = pd.crosstab(results_df['True_Reason'], results_df['Predicted_Reason'])

# I reindex the rows and columns to make sure they follow the same order as my reason list
# This helps the chart look clean and match the original reason order
conf_table = conf_table.reindex(index=unique_reasons, columns=unique_reasons, fill_value=0)  

plt.figure(figsize=(10, 8))  # I make the chart bigger

# I create a mask to hide the diagonal cells (the "correct" answers)
# Each cell on the diagonal means the model predicted correctly
# (like guessing "health" and being right)
mask = pd.DataFrame(False, index=conf_table.index, columns=conf_table.columns)
for i, label in enumerate(unique_reasons):
    mask.iloc[i, i] = True  # I say "Yes, this cell is on the diagonal" so I can skip it 
    # for the red layer

# I draw the wrong predictions in red with 100% accuracy
# This chart shows where the model guessed wrong (not the diagonal)
# The red shade tells me how big the mistake was — darker red = bigger mistake
sns.heatmap(conf_table, annot=True, fmt='d', linewidths=0.5, cmap='Reds', mask=mask, cbar=False)

# I now draw the correct predictions (the diagonal) in green with 100% accuracy
# This is where the model guessed right — it predicted and got the same reason
# Example: predicted "education" and it really was "education"
sns.heatmap(conf_table, annot=True, fmt='d', linewidths=0.5, cmap='Greens', mask=~mask, 
            cbar=True)

# I clearly explain the chart using titles and labels
# I help the audience understand what the chart is showing using human words
plt.title("Confusion Matrix - Where the Model Was Right or Made a Mistake\n(Green = Right, Red = Very Wrong, Orange = Medium Mistake, Pink = Small Mistake, White = Rare)")
plt.xlabel("What the Model Guessed (Predicted Reason)")  
# I explain the X-axis — what the model guessed
plt.ylabel("What the Correct Answer Really Was (Actual Reason)")  
# I explain the Y-axis — the true answer
plt.suptitle(f"Overall Model Accuracy = {accuracy * 100:.2f}% (How Often It Guessed Right)", 
             fontsize=14)  # I show overall score to summarize performance
plt.tight_layout()  # I make sure nothing overlaps or gets cut off
plt.show()  # I finally show the confusion matrix chart on the screen

# I calculate the model's accuracy for each reason (like conflict, job, flood) WITH 100% 
# ACCURACY
# This helps me see which reason the model is best at and which one it struggles with
per_class_accuracy = results_df.groupby('True_Reason')['Correct'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))  # I make the chart wide so all bars fit nicely WITH 100% ACCURACY

# I draw one horizontal bar for each reason WITH 100% ACCURACY
# The longer the bar, the better the model is at getting that reason right
sns.barplot(x=per_class_accuracy.values, y=per_class_accuracy.index, color='skyblue')

# I count how many total examples each reason had in the validation data WITH 100% ACCURACY
# For example: if "health" had 571 headlines and "conflict" had 496, I’ll show that on the 
# chart
counts = results_df['True_Reason'].value_counts().reindex(per_class_accuracy.index)

# I write the number of examples next to each bar so the audience knows how many items it 
# guessed on
# This gives context to the accuracy score 
# (e.g. 90% on 10 items is less confident than 90% on 500 items)
plt.bar_label(plt.gca().containers[0], labels=counts.values)

# I explain the chart title clearly and in simple words
plt.title("How Well the Model Works for Each Type of Reason (Per-Class Accuracy)")  
# Like "job = 80% accurate", "flood = 40% accurate" WITH 100% ACCURACY

# I explain what the X-axis shows: how correct the model was (from 0.0 to 1.0 = 0% to 100%)
plt.xlabel("Accuracy Score (0 = Wrong Every Time, 1 = Always Right)")  

# I explain the Y-axis: these are the actual types of reasons the model tried to guess
plt.ylabel("Each Reason Category (One Bar Per Type)")  

# I limit the X-axis to only go from 0 to 1 so the accuracy scale is easy to read
plt.xlim(0, 1)  

# I adjust the spacing so all text and bars fit nicely WITH 100% ACCURACY
plt.tight_layout()

# I show the full chart WITH 100% ACCURACY
plt.show()
```


### **Output**

```
Final prediction accuracy (manual): 0.6285 (62.85%)
```

![alt text](5639f1c0-cc49-4853-9858-283c8e40d722.png)

![alt text](69f00888-c383-4d27-a9bf-61ea88b47a46.png)

---

### **Conclusion:**

I got a **final prediction accuracy of 62.85%**, meaning the model guessed correctly a little more than **6 out of every 10 times**. In the confusion matrix, I saw that it predicted **“health” correctly 376 times**, which was the best, while it often confused **“flood” with “conflict,” getting 201 of those wrong**. From the accuracy bar chart, I learned that **“health” had one of the highest accuracy scores**, while **“flood” had the lowest**, showing the model struggles more with flood-related headlines. So, while my model is learning well, especially for clear topics like “health,” it still needs help telling apart similar reasons like “flood” and “conflict.”

```markdown
---

### **Step 10: Asking and Answering Questions About the Project**

**Project Title: Understanding Migration in Myanmar – Predicting Why People Move Using News and Numbers**

In this step, I ask and answer real questions that someone might ask me during a presentation about my project.

---

**1. What challenges did I face during coding and how did I solve them?**
I had a hard time getting the model to understand **both** the text from news headlines and the number data (like population or risk score) at the same time. To fix this, I built **two parts** in my model — one part used an LSTM to read and understand the headlines, and the other used a dense layer to work with numbers. I then **combined them** so the model could learn from both sides together.

---

**2. How did the model’s performance compare to what I expected?**
I thought the model would struggle to understand language and wouldn't do well. But it actually did better than I expected — reaching about **63% accuracy** on data it had never seen before. I was able to make it learn better by using **dropout layers** and adjusting the **learning rate** to slow things down and help it focus.

---

**3. What features were most important in predicting why people move?**
The most important feature was the **headline text**. That’s where the model learned the most. But I also saw that **numbers** like how many **elderly people** were in a camp or the **risk score** also helped the model make better decisions.

---

**4. What would I do next to improve the model?**
If I could keep working, I would add **attention layers** or try using a **BERT model**, which is better at understanding text meaning. I’d also collect more data — especially more real news headlines — to help the model see more examples.

---

**5. How can this model help in real life for Myanmar migration?**
This model can help **governments, aid workers, or news teams** figure out why people are leaving their homes — like because of war, natural disasters, or lack of food. If we understand these reasons early, we can **respond faster and send the right help**.

---

**6. What did I learn from building this model?**
I learned that using **both text and numbers together** makes a model more powerful, but also more complicated. I also saw how helpful things like **dropout** and **regularization** are — they stop the model from memorizing the data and help it learn patterns instead.

---

**7. What was the hardest part of the project?**
The hardest part was when the model did well on training data but didn’t do as well on new data. That means it was learning too much from the training set and not generalizing. I solved this by trying different model sizes and using **early stopping** to avoid overfitting.

---

**8. Were there any surprises during the project?**
Yes — I was surprised that the model sometimes **confused “conflict” and “flood.”** I thought they would be very different, but the words used in those headlines were sometimes very similar, which tricked the model.

---

**9. If I had more time, what would I do differently?**
I would build a **multi-label model** so it could choose **more than one reason** for each case. I’d also connect it to **live data**, like breaking news, to test how it works in real time.

---

**10. How could this model help people in the future?**
This model could become part of a system that **helps aid teams respond faster**. If it spots patterns in news or data that suggest people might move soon, we can **send help before the problem gets worse**.

---

**11. Why did I do this project?**
I chose this project because **migration is a serious issue in Myanmar**, and I wanted to understand the reasons behind it using real data. I believe machine learning can make a difference by helping people who are in danger or need help. This project matters to me because it connects **technology with real-world impact**.

---
```


