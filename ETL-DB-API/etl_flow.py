# -*- coding: utf-8 -*-
"""ETL_Flow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/cuauhtemocbe/Portfolio-Data-Scientist/blob/main/ETL-DB-API/ETL_Flow.ipynb

# ETL-API

# Setup

With the next command, will be install postgresql, and create the placeholder database
"""

# Installing postgresql server in linux
!sudo apt-get -y -qq update
!sudo apt-get -y -qq install postgresql
!sudo service postgresql start

# Setup a username and password
!sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
# Setup a database with name `placeholer`
!sudo -u postgres psql -U postgres -c 'DROP DATABASE IF EXISTS placeholder;'
# Creating the database placeholder
!sudo -u postgres psql -U postgres -c 'CREATE DATABASE placeholder;'

"""# 1. Creating table
- posts

For the tables comments, albums, photos, todos, users, will be created in the next steps in an automatic way.
"""

# Creating the table posts
!sudo -u postgres psql -U postgres -c 'CREATE TABLE IF NOT EXISTS posts (userId SERIAL PRIMARY KEY, id INT NOT NULL, title VARCHAR NOT NULL, body VARCHAR NOT NULL);'

"""#2. ETLs

In this section, I will be reading all the API endpoints using python libraries.

The first commands has a brief description of the steps to achieve the goal: read, transform and load the data into the database 'placeholders'.
"""

# Importing python libraries

# Requests allows to send HTTP request
import requests
# Pandas is a great tool to analyze and manage data tables
import pandas as pd
# sqlalchemy is a python sql toolkit
from sqlalchemy import create_engine

base_url = "https://jsonplaceholder.typicode.com"
resources_list = ["posts", "comments", "albums", "photos", "todos", "users"]
# Make the API requests to all resources in a iterative way
responses_list = list(map(lambda resource: requests.get(f"{base_url}/{resource}").json(),
                          resources_list))

# Checking the first response in data table (DataFrame object)
df_posts = pd.DataFrame(responses_list[0])
df_posts.head()

# The users data is more complicated, because some columns has
# a dict format, for example the column 'address' and 'company'
df_users = pd.DataFrame(responses_list[5])
df_users.head()

# With the next script we can check the data type
type(df_users['address'].iloc[0])

# It's necessary transform the data to unpack the dict into columns
col = "address"
new_columns = df_users.address.apply(pd.Series)
new_columns.columns = [f"{col}_{c}" for c in new_columns.columns]
new_columns

def dict_to_columns(df: pd.DataFrame):
  """Recursively transforms dict columns in a DataFrame into separate columns.
      Args: df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The modified DataFrame with expanded columns.
    """

  for col in df.columns:
    # Check if the column is a dict column
    if isinstance(df[col].iloc[0], dict):
      # Create a dataframe with the new columns from the dict column
      df_new_cols = df[col].apply(pd.Series)
      # Rename the new columns with the prefix of the origin column
      df_new_cols.columns = [f"{col}_{c}" for c in df_new_cols.columns]
      # Drop the column with dict data type
      df.drop(col, axis="columns", inplace=True)
      # Concatenate the new columns with the dataframe
      df = pd.concat([df, df_new_cols], axis="columns")
      # Recursively transform nested dict columns
      return dict_to_columns(df)

  return df

# Example
# The new columns added are: address_street, address_suite, address_city, address_zipcode,
# company_name, compay_catchPhrase, company_bs, address_geo_lat, address_geo_lng
dict_to_columns(df_users).head()

"""To load the data into 'placeholder' database I will use the next script:
```
engine = create_engine('postgresql://postgres:postgres@localhost:5432/placeholder')
pd.DataFrame(df_posts).to_sql('posts', engine)
```

In postgres we can create the table `posts` with the next query:
```
CREATE TABLE IF NOT EXISTS posts (
  userId SERIAL PRIMARY KEY,
  id INT UNSIGNED NOT NULL,
  title VARCHAR NOT NULL,
  body VARCHAR NOT NULL,
);
```
"""

# Dropping the table posts to clear the database
!sudo -u postgres psql -U postgres -c 'DROP TABLE IF EXISTS posts'

"""The next cmd summarize the previous steps in the main function etl."""

# elt.py
# Importing python libraries

# Requests allows to send HTTP request
import requests
# Pandas is a great tool to analyze and manage data tables
import pandas as pd
# sqlalchemy is a python sql toolkit
from sqlalchemy import create_engine

def etl():

  base_url = "https://jsonplaceholder.typicode.com"
  engine = create_engine('postgresql://postgres:postgres@localhost:5432/placeholder')

  # Resources or endpoints
  resources_list = ["posts", "comments", "albums", "photos", "todos", "users"]
  # Make the API requests to all resources in a iterative way
  # using a for loop to make easy to read the process
  for resource in resources_list:
    # [1] Get the data from API
    response = requests.get(f"{base_url}/{resource}")
    # [2] Convert to json format
    reponse_json = response.json()
    # [3] Convert to DataFrame object
    df = pd.DataFrame(reponse_json)
    # [4] Unpack the dict columns
    df = dict_to_columns(df)
    # [4] Load data into database
    df.to_sql(resource, engine, index=False)
    print(f"{resource} data was loaded successfully")

etl()

"""#3. Data Analysis

## a) Retrieve the userId that made most comments on all the Posts:

Answer: All the userId has the same number of comments.
"""

from sqlalchemy import create_engine

# Connect to the PostgreSQL database
engine = create_engine('postgresql://postgres:postgres@localhost:5432/placeholder')
connection = engine.raw_connection()

query = "SELECT * FROM posts"
df = pd.read_sql(query, connection)
df.head()

# If each row is a comment made by a userId, we can group by userId and count
df.groupby(["userId"])[["userId"]].count()

"""## b) Retrieve the number of comments per Post

In this exercise, I will be using de data `comments`, to get the number of comments for each postId
"""

query = "SELECT * FROM comments"
df_comments = pd.read_sql(query, connection)
df_comments.head(5)

# If each row is a comment per post (postId), we can group by postId and count
df_comments.groupby(["postId"])[["postId"]].count()

"""All the posts has the same number of comments.

## c) Retrieve the longest post comment made on all the posts
"""

query = "SELECT * FROM posts"
df_post = pd.read_sql(query, connection)
df_posts.head()

# Count the number of words to get the lognest post comment
df_posts['total_words'] = df['body'].str.split().str.len()

# Sort the comments by total words
df_posts.sort_values("total_words", ascending=False)

"""The longest post comment is made by userId 4 and comment id equal to 36."""

# Close the connection
connection.close()