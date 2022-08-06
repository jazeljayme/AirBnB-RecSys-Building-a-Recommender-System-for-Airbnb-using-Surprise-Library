import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import geopandas
import surprise
from surprise import (Reader, Dataset, KNNWithMeans, SVD)
from surprise import accuracy
from surprise.model_selection import cross_validate, GridSearchCV

from afinn import Afinn

import glob
import warnings
warnings.filterwarnings('ignore')


def get_listings():
    """Return the dataframe that contains the listings."""

    # load all files (to be safe!)
    files = glob.glob('/mnt/data/public/insideairbnb/' +
                      'data.insideairbnb.com/united-kingdom/' +
                      'england/london/2021*/data/listings.csv.gz')
    listings = pd.concat([pd.read_csv(f) for f in files])
    listings_orig_shape = listings.shape

    # create a unique identifier to identify and remove duplicates
    listings['code'] = listings['id'].apply(str) + '-' + listings[
                        'host_id'].apply(str)

    # remove duplicates and drop unique identifier and nans
    listings = listings.drop_duplicates(subset=['code'])
    listings = listings.drop(['code'], axis=1)
    return listings


def get_reviews():
    """Return the dataframe that contains the reviews."""

    # load all files (to be safe!)
    files = glob.glob('/mnt/data/public/insideairbnb/data.insideairbnb.com' +
                      '/united-kingdom/england/london/2021*/' +
                      'data/reviews.csv.gz')
    reviews = pd.concat([pd.read_csv(f) for f in files])
    reviews_orig_shape = reviews.shape

    # create a unique identifier to identify and remove duplicates
    reviews['code'] = reviews['listing_id'].apply(str) + '-' + reviews[
                        'id'].apply(str) + '-' + reviews[
                        'date'].apply(str) + '-' + reviews[
                        'reviewer_id'].apply(str)

    # remove duplicates and drop unique identifier and nans
    reviews = reviews.drop_duplicates(subset=['code'])
    reviews = reviews.drop(['code'], axis=1).dropna(subset=['comments'])
    return reviews


def extract_clean_listing_df(df_listings):
    """Return a cleaned dataset by sorting and removing duplicates."""

    # extracting only useful columns
    df1_listings = df_listings.dropna()
    df1_listings.sort_values(['id', 'last_scraped'], ascending=[True, True],
                             inplace=True)
    df1_listings.rename(columns={'id': 'listing_id'}, inplace=True)
    # Find duplicated listings
    dups_listings = df1_listings[df1_listings.duplicated(
                                 subset='listing_id')]['listing_id'].unique()
    print('Number of listings with duplicates: ', len(dups_listings))

    # drop duplicated listing_id and keep the recent record
    df2_listings = df1_listings.drop_duplicates(subset='listing_id',
                                                keep='last')
    return df2_listings


def preprocessing_df(df_listings, df_reviews):
    """Return a datafame which resulted from merging `df_listings` and
    `df_reviews` with additional filtering.

    Keep listings that were rated at least 5 times by guests and keep
    guests who rated at least 10 times in any of the listings.
    """

    # merging the two dataframes
    cols2 = ['listing_id', 'reviewer_id', 'comments']
    df = df_listings.merge(df_reviews[cols2], on=['listing_id'])

    # Filtering
    min_listing_ratings = 10  # a listing was rated at least
    min_guest_ratings = 10  # a guest rated a place at least

    df_new = df.groupby("listing_id").filter(
                        lambda x: x['listing_id'].count() >=
                        min_listing_ratings)
    df_new = df_new.groupby("reviewer_id").filter(
                            lambda x: x['reviewer_id'].count() >=
                            min_guest_ratings)
    return df_new


def plot_Dist_sentiment_Score(df, bins, col='sentiment_score', f_num=2):
    """Return the histogram plot of the sentiment scores."""

    plt.figure(figsize=(8, 5))
    plt.title('Figure '+str(f_num)+'. Distribution on the sentiment ratings' +
              ' from Afinn model', fontsize=16)

    plt.hist(df[col], bins=bins, color='#FF5A60')
    plt.ylabel('Number of listings with X rating', fontsize=14)
    plt.xlabel('Sentiment Ratings', fontsize=14)
    plt.show()


def word_count(text_string):
    """"Calculate the number of words in a string."""
    return len(text_string.split())


def get_best_param_SVD(data):
    """Return the optimal `n_factors` for the SVD RecSys algorithm.

    Parameters
    -----------
        data : the dataframe with ratings by guest per listing
               with Dataset.load_from_df
    """

    param_grid = {'n_factors': [10, 20, 30, 40, 50, 80, 100]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
    gs.fit(data)
    best_param = gs.best_params['rmse']
    return best_param


def recsys_model(model, trainset, testset):
    """Return the prediction and RMSE of the recommender system model.
    Parameters
    -----------
    model    : model that will be trained
    trainset : the trainset returned by the surprise.trainset.Trainset
    testset  : the testset the model will be tested.
    """

    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    return predictions, rmse


def get_top_n(predictions, reviewer_id, df, n=10):
    """Return the top N similar guests for a `reviewer_id`.

    Parameters
    -----------
        predictions : list
                    Predictions of the test data using SVD method
                    with surprise.

        reviewer_id : int
                    The guest ID in the AirBnb database.

        df          : Pandas DataFrame
                    The dataframe that stores the all information including
                    `listing_id`, `reviewer_id`, `adj_sentiment_score`.

        n           : int
                    The number of top recommendations needed.
    """

    top_n = defaultdict(list)
    # mapping the predictions to each guest
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sorting the predictions for each guest and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # DataFrame with predictions
    preds_df = pd.DataFrame([(id, pair[0], pair[1]) for id, row in
                             top_n.items() for pair in row],
                            columns=["reviewer_id",
                                     "listing_id",
                                     "adj_sentiment_score"])

    # Return pred_usr, i.e. top N recommended listing
    listing_df = df[['listing_id',
                     'latitude',
                     'longitude',
                     'host_id',
                     'property_type',
                     'listing_url']]
    recom_listing = preds_df[preds_df["reviewer_id"] == reviewer_id][
                    'listing_id']
    pred_usr = listing_df[listing_df['listing_id'].isin(
               recom_listing)].drop_duplicates()

    # Return hist_usr, i.e. top N historically rated listings
    hist_usr = df[df.reviewer_id == reviewer_id].sort_values(
                        "adj_sentiment_score", ascending=False)

    return hist_usr, pred_usr


def get_recommended_listings(predictions, n=1000):
    """Return the recommended listings and how many times the listing
    is recommended among all guests.

    Recommended listings are those that belong in the top 1000 recommendation.

    Parameters
    -----------
    predictions : list
                Predictions of the test data using any RecSys method
                with surprise.
    n           : int
                The number of top recommendations needed.
    """

    top_n = defaultdict(list)
    # mapping the predictions to each guest
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # DataFrame with predictions
    preds_df = pd.DataFrame([(id, pair[0], pair[1]) for id, row in
                             top_n.items() for pair in row],
                            columns=["reviewer_id",
                                     "listing_id",
                                     "adj_sentiment_score"])

    listing_counts = pd.DataFrame(preds_df['listing_id'].value_counts(
                                  )).reset_index()
    listing_counts.rename(columns={'index': 'listing_id',
                                   'listing_id': 'counts'}, inplace=True)
    return listing_counts


def plot_sentiments_listings(df, listings, f_num=10):
    """Return the two plots that contain the sentiment score rating for those
    NOT recommended by the RecSys and those that are recommended.

    Parameters
    -----------
        df       : pandas DataFrame
                  The dataframe containing the raw dataset.

        listings : list
                  The list containing the recommended listings from the model.
    """

    not_recommended = df[~(df['listing_id'].isin(listings))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    fig.subplots_adjust(top=0.8)
    plt.suptitle('Figure '+str(f_num)+'. The distribution of sentiment '
                 'scores for the recommended and not recommended listings',
                 fontsize=20)
    ax1.set_title('NOT recommended by the Recommender System', fontsize=16)
    ax1.hist(df[df.listing_id.isin(not_recommended.listing_id
                                   .unique())]['adj_sentiment_score'],
             color='#767676')

    ax2.set_title('Recommended by the Recommender System', fontsize=16)
    ax2.hist(df[df.listing_id.isin(listings
                                   .unique())]['adj_sentiment_score'],
             color='#FF5A60')


def maps(df, data=None):
    """Returns an interactive Folium map of the distribution of
    `adj_sentiment_score` and `property_count`.

    Parameters
    -----
    listings : csv
        Contains the Airbnb listings, specifically `reviewer_id`,
        `listing_id`, `adj_sentiment_score`, `latitude`, `longitude`,
        `property_type`, and `host_id`.

    Returns
    -----
    Folium map
        An interactive map showing either the distribution of
        `adj_sentiment_score` or `property_count`.

    """
    # read files
    listings = df.drop_duplicates(subset='listing_id')
    msoas = geopandas.read_file('statistical-gis-boundaries-london' +
                                '/ESRI/MSOA_2004_London_High_Resolution.shp')
    boroughs = geopandas.read_file('statistical-gis-boundaries-london/' +
                                   'ESRI/London_Borough_Excluding_MHW.shp')

    # get points and assign to geom
    pts = geopandas.points_from_xy(listings['longitude'],
                                   listings['latitude'])
    x = ['reviewer_id', 'listing_id', 'adj_sentiment_score', 'property_type',
         'host_id']
    geo_listings = geopandas.GeoDataFrame(listings[x].assign(geometry=pts),
                                          crs='EPSG:4326')

    # match with London MSOA codes and names
    db = geopandas.sjoin(geo_listings,
                         msoas[['geometry',
                                'MSOA_CODE',
                                'MSOA_NAME']].to_crs(geo_listings.crs),
                         how='left')
    g = db.groupby('MSOA_CODE')
    msoas_abb = g.mean().drop('index_right', axis=1)
    msoas_abb['property_count'] = g.size()
    msoas_abb = geopandas.GeoDataFrame(msoas_abb.join(msoas.set_index(
                                       'MSOA_CODE')[['geometry']]),
                                       crs=msoas.crs)
    msoa_cents = geopandas.GeoDataFrame({'MSOA11CD': msoas_abb.index,
                                         'geometry': msoas_abb.centroid},
                                        crs=msoas_abb.crs)
    msoa2borough = geopandas.sjoin(msoa_cents, boroughs[['NAME',
                                                         'GSS_CODE',
                                                         'geometry']]
                                   .to_crs(msoas_abb.crs), how='left')
    abb = msoas_abb.reset_index().join(msoa2borough[['NAME',
                                                     'GSS_CODE']],
                                       on='MSOA_CODE').rename({'NAME':
                                                               'BOROUGH'},
                                                              axis=1)

    # return whichever is required
    if data == 'property_count':
        return abb.drop(['adj_sentiment_score',
                         'reviewer_id',
                         'host_id',
                         'listing_id'],
                        axis=1).explore(tiles='CartoDB positron',
                                        column='property_count',
                                        scheme='quantiles',
                                        cmap='Set1_r')

    elif data == 'sentiment_score':
        return abb.drop(['property_count',
                         'reviewer_id',
                         'host_id',
                         'listing_id'],
                        axis=1).explore(tiles='CartoDB positron',
                                        column='adj_sentiment_score',
                                        scheme='quantiles',
                                        cmap='Set1_r')

    if data is None:
        return geo_listings.explore(tiles='CartoDB positron',
                                    column='adj_sentiment_score',
                                    marker_type='circle_marker',
                                    marker_kwds={'radius': 2,
                                                 'fill': True},
                                    cmap='Set1_r')


def recommender_maps(historical, svd, itemBased, userBased):
    """Returns an interactive Folium map of the list of recommendations from
    a user's historical data; and SVD, item-based, and user-based recommender
    systems.

    Parameters
    -----
    historical, svd, itemBased, userBased : data frame
        Contains the Airbnb listings, specifically `reviewer_id`,
        `listing_id`, `adj_sentiment_score`, `latitude`, `longitude`,
        `property_type`, and `host_id`.

    Returns
    -----
    Folium map
        An interactive map with all the recommendations plotted in the area.

    """
    historical['type'] = 'historical'
    svd['type'] = 'svd'
    itemBased['type'] = 'itemBased'
    userBased['type'] = 'userBased'

    dfs = [historical, svd, itemBased, userBased]
    out = []
    for df in dfs:
        x = df.drop_duplicates(subset=['listing_id'], keep='first')[:10]
        out.append(x)

    concat_maps = pd.concat(out)
    x = ['adj_sentiment_score', 'listing_id', 'property_type', 'type']
    pts = geopandas.points_from_xy(concat_maps["longitude"],
                                   concat_maps["latitude"])
    geo_listings = geopandas.GeoDataFrame(concat_maps[x].assign(geometry=pts),
                                          crs="EPSG:4326")

    return geo_listings.explore(tiles='CartoDB positron',
                                marker_type='circle_marker',
                                marker_kwds={'radius': 5,
                                             'fill': True},
                                categorical=True,
                                column='type',
                                cmap='Set1_r')
    return out
