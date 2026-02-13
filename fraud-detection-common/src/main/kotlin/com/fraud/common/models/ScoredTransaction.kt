package com.fraud.common.models

import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Represents a transaction that has been scored for fraud risk by the ML model.
 * 
 * This data class extends the Transaction concept by adding fraud detection results,
 * including the fraud score (probability) and the timestamp when scoring occurred.
 * Used in the inference pipeline to store and alert on high-risk transactions.
 * 
 * @property transactionId Unique transaction identifier (corresponds to Transaction.id)
 * @property timestamp Unix timestamp (milliseconds) when the transaction occurred
 * @property amount Transaction amount in currency units
 * @property merchantCategory Category of the merchant (e.g., "retail", "online", "restaurant")
 * @property features Map of feature names to values (V1-V28 from Kaggle dataset, plus Time and Amount)
 * @property fraudScore Fraud probability score from 0.0 (legitimate) to 1.0 (fraudulent)
 * @property scoringTimestamp Unix timestamp (milliseconds) when the fraud score was calculated
 */
data class ScoredTransaction(
    @JsonProperty("transactionId")
    val transactionId: String,
    
    @JsonProperty("timestamp")
    val timestamp: Long,
    
    @JsonProperty("amount")
    val amount: Double,
    
    @JsonProperty("merchantCategory")
    val merchantCategory: String,
    
    @JsonProperty("features")
    val features: Map<String, Double>,
    
    @JsonProperty("fraudScore")
    val fraudScore: Double,
    
    @JsonProperty("scoringTimestamp")
    val scoringTimestamp: Long
) {
    init {
        require(fraudScore in 0.0..1.0) {
            "Fraud score must be between 0.0 and 1.0, but was $fraudScore"
        }
    }
    
    /**
     * Determines if this transaction is considered high-risk based on the fraud score.
     * High-risk transactions have a fraud score >= 0.8 and trigger alerts.
     */
    @JsonIgnore
    fun isHighRisk(): Boolean = fraudScore >= 0.8
    
    /**
     * Determines if this transaction is considered medium-risk based on the fraud score.
     * Medium-risk transactions have a fraud score between 0.5 and 0.8.
     */
    @JsonIgnore
    fun isMediumRisk(): Boolean = fraudScore >= 0.5 && fraudScore < 0.8
    
    /**
     * Determines if this transaction is considered low-risk based on the fraud score.
     * Low-risk transactions have a fraud score < 0.5.
     */
    @JsonIgnore
    fun isLowRisk(): Boolean = fraudScore < 0.5
}
