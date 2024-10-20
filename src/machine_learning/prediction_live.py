import streamlit as st
import pandas as pd


def predict_success(X_live, best_features, ml_pipe_fe, ml_pipe_model):

    # Handle edge case. If 0 embryos were created or transferred,
    # predict failure.
    if X_live["Embryos transferred"].iloc[0] == "0":
        statement = (
            f"### There is a 100% probability that the treatment **will"
            f" not be successful** "
            f"as no embryos were transferred."
        )
        st.write(statement)
        return 0

    # If embryos were not created, but they might have been donated,
    # let the model predict.

    # Subset features related to this pipeline
    X_live_ivf = X_live.filter(best_features)

    # Apply preprocessing pipeline to the live data
    X_live_ivf_dc_fe = ml_pipe_fe.transform(X_live_ivf)

    # Extract expected feature names directly from the model
    try:
        expected_feature_names = ml_pipe_model.feature_names_in_
    except AttributeError:
        expected_feature_names = best_features

    # Create a DataFrame from transformed data and reindex with
    # expected features
    X_live_ivf_dc_fe_best = pd.DataFrame(
        X_live_ivf_dc_fe, columns=expected_feature_names
    )

    # Reindex to ensure exact match of columns as per the trained model
    X_live_ivf_dc_fe_best = X_live_ivf_dc_fe_best.reindex(
        columns=expected_feature_names, fill_value=0
    )

    # Print the transformed data to check
    print(X_live_ivf_dc_fe_best.head())

    # Predict using the model
    success_prediction = ml_pipe_model.predict(X_live_ivf_dc_fe_best)
    success_prediction_proba = ml_pipe_model.predict_proba(
        X_live_ivf_dc_fe_best
        )

    # Get probability of success (class 1)
    success_prob = success_prediction_proba[0, 1] * 100
    success_result = "will" if success_prediction == 1 else "will not"

    # Additional statement for donated embryos
    if (
        X_live["Total embryos created"].iloc[0] == "0"
        and X_live["Embryos transferred"].iloc[0] != "0"
    ):
        additional_statement = " for donated embryos"
    else:
        additional_statement = ""

    # Display results with the additional statement for donated embryos
    statement = (
        f"### There is a {success_prob.round(1)}% probability "
        f"that the treatment **{success_result} be successful**"
        f"{additional_statement}."
    )
    st.write(statement)

    return success_prediction
