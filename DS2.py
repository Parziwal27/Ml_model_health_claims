from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load your trained model and scaler
try:
    logger.info("Loading model and scaler...")
    model = joblib.load('lgbm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Define the expected features (all except 'InscClaimAmtReimbursed')
EXPECTED_FEATURES = [
    'BeneID', 'ClaimID', 'Provider', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
    'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
    'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
    'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
    'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
    'DeductibleAmtPaid', 'ClmAdmitDiagnosisCode', 'Gender', 'Race', 'RenalDiseaseIndicator',
    'State', 'County', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
    'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
    'ChronicCond_stroke', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'IsDeceased', 'Age', 'ClaimDuration',
    'AgeAtClaim', 'ClaimLengthCategory', 'AgeCategory', 'NumChronicConditions', 'NumDiagnoses',
    'NumProcedures', 'AgeChronicInteraction', 'DurationProcedureInteraction'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        # Get the data from the POST request
        try:
            data = request.json['features']
            logger.info(f"Received data: {data}")
        except KeyError:
            logger.error("No 'features' key in JSON data")
            return jsonify({'error': 'No features provided in the request'}), 400
        except Exception as e:
            logger.error(f"Error parsing request data: {str(e)}")
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Ensure all required features are present
        missing_features = [f for f in EXPECTED_FEATURES if f not in data]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return jsonify({'error': f'Missing required features: {", ".join(missing_features)}'}), 400
        
        # Create feature array, converting each feature to float
        features = []
        for feature in EXPECTED_FEATURES:
            try:
                features.append(float(data[feature]))
            except ValueError:
                logger.error(f"Invalid value for feature {feature}: {data[feature]}")
                return jsonify({'error': f'Invalid value for {feature}. Expected float.'}), 400
        
        logger.info("Features extracted successfully")
        
        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        try:
            scaled_features = scaler.transform(features_array)
            logger.info("Features scaled successfully")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return jsonify({'error': 'Error scaling features'}), 500
        
        # Make prediction
        try:
            prediction = model.predict(scaled_features)
            logger.info(f"Prediction made: {prediction}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({'error': 'Error making prediction'}), 500
        
        response = {'prediction': float(prediction[0])}
        logger.info(f"Sending response: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set to False in production
