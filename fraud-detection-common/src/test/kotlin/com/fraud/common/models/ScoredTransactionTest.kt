package com.fraud.common.models

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll

class ScoredTransactionTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()

    test("ScoredTransaction should be created with valid fraud score") {
        val scoredTransaction = ScoredTransaction(
            transactionId = "txn-001",
            timestamp = 1705334400000L,
            amount = 149.62,
            merchantCategory = "retail",
            features = mapOf("V1" to -1.36, "Amount" to 149.62),
            fraudScore = 0.0234,
            scoringTimestamp = 1705334401000L
        )

        scoredTransaction.transactionId shouldBe "txn-001"
        scoredTransaction.fraudScore shouldBe 0.0234
        scoredTransaction.scoringTimestamp shouldBe 1705334401000L
    }

    test("ScoredTransaction should reject fraud score below 0.0") {
        val exception = shouldThrow<IllegalArgumentException> {
            ScoredTransaction(
                transactionId = "txn-002",
                timestamp = 1705334400000L,
                amount = 100.0,
                merchantCategory = "retail",
                features = emptyMap(),
                fraudScore = -0.1,
                scoringTimestamp = 1705334401000L
            )
        }
        
        exception.message shouldContain "Fraud score must be between 0.0 and 1.0"
        exception.message shouldContain "-0.1"
    }

    test("ScoredTransaction should reject fraud score above 1.0") {
        val exception = shouldThrow<IllegalArgumentException> {
            ScoredTransaction(
                transactionId = "txn-003",
                timestamp = 1705334400000L,
                amount = 100.0,
                merchantCategory = "retail",
                features = emptyMap(),
                fraudScore = 1.5,
                scoringTimestamp = 1705334401000L
            )
        }
        
        exception.message shouldContain "Fraud score must be between 0.0 and 1.0"
        exception.message shouldContain "1.5"
    }

    test("ScoredTransaction should accept fraud score at boundary 0.0") {
        val scoredTransaction = ScoredTransaction(
            transactionId = "txn-004",
            timestamp = 1705334400000L,
            amount = 100.0,
            merchantCategory = "retail",
            features = emptyMap(),
            fraudScore = 0.0,
            scoringTimestamp = 1705334401000L
        )

        scoredTransaction.fraudScore shouldBe 0.0
    }

    test("ScoredTransaction should accept fraud score at boundary 1.0") {
        val scoredTransaction = ScoredTransaction(
            transactionId = "txn-005",
            timestamp = 1705334400000L,
            amount = 100.0,
            merchantCategory = "retail",
            features = emptyMap(),
            fraudScore = 1.0,
            scoringTimestamp = 1705334401000L
        )

        scoredTransaction.fraudScore shouldBe 1.0
    }

    test("isHighRisk should return true for fraud score >= 0.8") {
        val highRisk = ScoredTransaction(
            transactionId = "txn-006",
            timestamp = 1705334400000L,
            amount = 2500.0,
            merchantCategory = "online",
            features = emptyMap(),
            fraudScore = 0.8912,
            scoringTimestamp = 1705334401000L
        )

        highRisk.isHighRisk() shouldBe true
        highRisk.isMediumRisk() shouldBe false
        highRisk.isLowRisk() shouldBe false
    }

    test("isHighRisk should return true for fraud score exactly 0.8") {
        val highRisk = ScoredTransaction(
            transactionId = "txn-007",
            timestamp = 1705334400000L,
            amount = 1000.0,
            merchantCategory = "online",
            features = emptyMap(),
            fraudScore = 0.8,
            scoringTimestamp = 1705334401000L
        )

        highRisk.isHighRisk() shouldBe true
    }

    test("isMediumRisk should return true for fraud score between 0.5 and 0.8") {
        val mediumRisk = ScoredTransaction(
            transactionId = "txn-008",
            timestamp = 1705334400000L,
            amount = 500.0,
            merchantCategory = "retail",
            features = emptyMap(),
            fraudScore = 0.65,
            scoringTimestamp = 1705334401000L
        )

        mediumRisk.isHighRisk() shouldBe false
        mediumRisk.isMediumRisk() shouldBe true
        mediumRisk.isLowRisk() shouldBe false
    }

    test("isLowRisk should return true for fraud score < 0.5") {
        val lowRisk = ScoredTransaction(
            transactionId = "txn-009",
            timestamp = 1705334400000L,
            amount = 50.0,
            merchantCategory = "restaurant",
            features = emptyMap(),
            fraudScore = 0.0234,
            scoringTimestamp = 1705334401000L
        )

        lowRisk.isHighRisk() shouldBe false
        lowRisk.isMediumRisk() shouldBe false
        lowRisk.isLowRisk() shouldBe true
    }

    test("ScoredTransaction should serialize to JSON correctly") {
        val scoredTransaction = ScoredTransaction(
            transactionId = "txn-010",
            timestamp = 1705334400000L,
            amount = 149.62,
            merchantCategory = "retail",
            features = mapOf("V1" to -1.36, "Amount" to 149.62),
            fraudScore = 0.0234,
            scoringTimestamp = 1705334401000L
        )

        val json = objectMapper.writeValueAsString(scoredTransaction)
        
        json shouldNotBe null
        json.contains("\"transactionId\":\"txn-010\"") shouldBe true
        json.contains("\"fraudScore\":0.0234") shouldBe true
        json.contains("\"scoringTimestamp\":1705334401000") shouldBe true
    }

    test("ScoredTransaction should deserialize from JSON correctly") {
        val json = """
            {
                "transactionId": "txn-011",
                "timestamp": 1705334400000,
                "amount": 2500.00,
                "merchantCategory": "online",
                "features": {
                    "Time": 100.0,
                    "V1": 2.45,
                    "Amount": 2500.00
                },
                "fraudScore": 0.8912,
                "scoringTimestamp": 1705334401000
            }
        """.trimIndent()

        val scoredTransaction: ScoredTransaction = objectMapper.readValue(json)

        scoredTransaction.transactionId shouldBe "txn-011"
        scoredTransaction.timestamp shouldBe 1705334400000L
        scoredTransaction.amount shouldBe 2500.00
        scoredTransaction.merchantCategory shouldBe "online"
        scoredTransaction.fraudScore shouldBe 0.8912
        scoredTransaction.scoringTimestamp shouldBe 1705334401000L
        scoredTransaction.features.size shouldBe 3
        scoredTransaction.isHighRisk() shouldBe true
    }

    test("ScoredTransaction should support data class copy") {
        val original = ScoredTransaction(
            transactionId = "txn-012",
            timestamp = 1705334400000L,
            amount = 100.0,
            merchantCategory = "retail",
            features = mapOf("V1" to 1.0),
            fraudScore = 0.5,
            scoringTimestamp = 1705334401000L
        )

        val copied = original.copy(fraudScore = 0.9)

        copied.transactionId shouldBe original.transactionId
        copied.timestamp shouldBe original.timestamp
        copied.amount shouldBe original.amount
        copied.fraudScore shouldBe 0.9
        copied.isHighRisk() shouldBe true
    }

    // Feature: fraud-detection-ml-pipeline, Property 10: Fraud Score Range
    test("Property 10: Fraud Score Range - valid scores in [0.0, 1.0] should be accepted") {
        checkAll(100, Arb.double(0.0, 1.0)) { fraudScore ->
            // For any fraud score in the valid range [0.0, 1.0]
            val scoredTransaction = ScoredTransaction(
                transactionId = "txn-prop-${fraudScore.hashCode()}",
                timestamp = System.currentTimeMillis(),
                amount = 100.0,
                merchantCategory = "test",
                features = emptyMap(),
                fraudScore = fraudScore,
                scoringTimestamp = System.currentTimeMillis()
            )
            
            // The fraud score should be stored correctly
            scoredTransaction.fraudScore shouldBe fraudScore
            // And should be within the valid range
            (scoredTransaction.fraudScore >= 0.0) shouldBe true
            (scoredTransaction.fraudScore <= 1.0) shouldBe true
        }
    }

    // Feature: fraud-detection-ml-pipeline, Property 10: Fraud Score Range
    test("Property 10: Fraud Score Range - invalid scores outside [0.0, 1.0] should be rejected") {
        // Test scores below 0.0
        checkAll(50, Arb.double(-1000.0, -0.0001)) { invalidScore ->
            shouldThrow<IllegalArgumentException> {
                ScoredTransaction(
                    transactionId = "txn-invalid-${invalidScore.hashCode()}",
                    timestamp = System.currentTimeMillis(),
                    amount = 100.0,
                    merchantCategory = "test",
                    features = emptyMap(),
                    fraudScore = invalidScore,
                    scoringTimestamp = System.currentTimeMillis()
                )
            }
        }
        
        // Test scores above 1.0
        checkAll(50, Arb.double(1.0001, 1000.0)) { invalidScore ->
            shouldThrow<IllegalArgumentException> {
                ScoredTransaction(
                    transactionId = "txn-invalid-${invalidScore.hashCode()}",
                    timestamp = System.currentTimeMillis(),
                    amount = 100.0,
                    merchantCategory = "test",
                    features = emptyMap(),
                    fraudScore = invalidScore,
                    scoringTimestamp = System.currentTimeMillis()
                )
            }
        }
    }
})
