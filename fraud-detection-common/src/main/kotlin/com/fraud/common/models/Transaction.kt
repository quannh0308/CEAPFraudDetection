package com.fraud.common.models

import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Represents a bank transaction record containing features for fraud detection.
 * 
 * This data class models transactions from the Kaggle Credit Card Fraud Detection dataset,
 * which contains anonymized features (V1-V28) derived from PCA transformation.
 * 
 * @property id Unique transaction identifier
 * @property timestamp Unix timestamp (milliseconds) when the transaction occurred
 * @property amount Transaction amount in currency units
 * @property merchantCategory Category of the merchant (e.g., "retail", "online", "restaurant")
 * @property features Map of feature names to values (V1-V28 from Kaggle dataset, plus Time and Amount)
 */
data class Transaction(
    @JsonProperty("id")
    val id: String,
    
    @JsonProperty("timestamp")
    val timestamp: Long,
    
    @JsonProperty("amount")
    val amount: Double,
    
    @JsonProperty("merchantCategory")
    val merchantCategory: String,
    
    @JsonProperty("features")
    val features: Map<String, Double>
)
