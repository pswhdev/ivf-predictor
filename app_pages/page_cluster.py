import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_ifv_treatment_data, load_pickle_file


def page_cluster_body():

    # load cluster analysis files and pipeline
    version = "v1"
    cluster_pipe = load_pickle_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl"
    )
    best_features = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/best_features_clusters.csv.gz"
    )
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/" f"clusters_silhouette.png"
    )
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/"
        f"{version}/features_define_cluster.png"
    )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/" "clusters_profile.csv.gz"
    )
    cluster_features = pd.read_csv(
        f"outputs/ml_pipeline/" f"cluster_analysis/{version}/" "TrainSet.csv.gz"
    ).columns.to_list()

    # dataframe for cluster_distribution_per_variable()
    df_success_vs_clusters = load_ifv_treatment_data().filter(
        ["Live birth occurrence"], axis=1
    )
    df_success_vs_clusters["Clusters"] = cluster_pipe["model"].labels_

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        """
        The cluster analysis pipeline was trained using the KMeans
        algorithm but proved to be ineffective in this scenario, as
        indicated by a low silhouette score of 0.15. Initially, the
        model was trained with 4 clusters, and after reducing the
        number of variables, the model was retrained again with 4 clusters.
        
        However, the results remained suboptimal, with an average
        silhouette score below 0.4 and only two clusters. One of them
        showed points widely spread between -0.4 and 0.4,
        indicating poor clustering performance.
        """
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### Features used in the model and their importance")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    st.write("#### Cluster Profile")
    statement = """
        The cluster analysis did not yield useful insights for associating the
        clusters with the target variable, as all groups exhibited a similar
        distribution of the target variable. However, this analysis was
        valuable as it highlighted the limited predictive power of the data
        within this specific dataset.
        """
    st.info(statement)

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)



