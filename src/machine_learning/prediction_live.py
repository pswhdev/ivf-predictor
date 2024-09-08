import streamlit as st
import pandas as pd


def predict_success(X_live, best_features, ml_pipe_fe, ml_pipe_model):

    # Subset features related to this pipeline
    X_live_ivf = X_live.filter(best_features)

    # Apply preprocessing pipeline to the live data
    X_live_ivf_dc_fe = ml_pipe_fe.transform(X_live_ivf)

    # Extract expected feature names directly from the model or
    # manually set them if needed
    try:
        # Attempt to extract feature names from the model pipeline
        expected_feature_names = ml_pipe_model.feature_names_in_
    except AttributeError:
        # Set manually or retrieve from a known source if not
        # accessible from the model
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

    # Predict using the model
    success_prediction = ml_pipe_model.predict(X_live_ivf_dc_fe_best)
    success_prediction_proba = ml_pipe_model.predict_proba(
        X_live_ivf_dc_fe_best
        )

    # Calculate success probability
    success_prob = success_prediction_proba[0, success_prediction][0] * 100
    success_result = "will" if success_prediction == 1 else "will not"

    # Display results
    statement = (
        f"### There is a {success_prob.round(1)}% probability "
        f"that the treatment **{success_result} be successful**."
    )
    st.write(statement)

    return success_prediction
