package com.fraud.common.models

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe

class TransactionTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()

    test("Transaction should be created with all required fields") {
        val transaction = Transaction(
            id = "txn-001",
            timestamp = 1705334400000L,
            amount = 149.62,
            merchantCategory = "retail",
            features = mapOf(
                "Time" to 0.0,
                "V1" to -1.3598071336738,
                "V2" to -0.0727811733098497,
                "Amount" to 149.62
            )
        )

        transaction.id shouldBe "txn-001"
        transaction.timestamp shouldBe 1705334400000L
        transaction.amount shouldBe 149.62
        transaction.merchantCategory shouldBe "retail"
        transaction.features.size shouldBe 4
        transaction.features["V1"] shouldBe -1.3598071336738
    }

    test("Transaction should serialize to JSON correctly") {
        val transaction = Transaction(
            id = "txn-002",
            timestamp = 1705334500000L,
            amount = 2500.00,
            merchantCategory = "online",
            features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
        )

        val json = objectMapper.writeValueAsString(transaction)
        
        json shouldNotBe null
        json.contains("\"id\":\"txn-002\"") shouldBe true
        json.contains("\"timestamp\":1705334500000") shouldBe true
        json.contains("\"amount\":2500.0") shouldBe true
        json.contains("\"merchantCategory\":\"online\"") shouldBe true
    }

    test("Transaction should deserialize from JSON correctly") {
        val json = """
            {
                "id": "txn-003",
                "timestamp": 1705334600000,
                "amount": 75.50,
                "merchantCategory": "restaurant",
                "features": {
                    "Time": 200.0,
                    "V1": 1.23,
                    "V2": -0.45,
                    "Amount": 75.50
                }
            }
        """.trimIndent()

        val transaction: Transaction = objectMapper.readValue(json)

        transaction.id shouldBe "txn-003"
        transaction.timestamp shouldBe 1705334600000L
        transaction.amount shouldBe 75.50
        transaction.merchantCategory shouldBe "restaurant"
        transaction.features.size shouldBe 4
        transaction.features["V1"] shouldBe 1.23
        transaction.features["V2"] shouldBe -0.45
    }

    test("Transaction should handle empty features map") {
        val transaction = Transaction(
            id = "txn-004",
            timestamp = 1705334700000L,
            amount = 0.0,
            merchantCategory = "test",
            features = emptyMap()
        )

        transaction.features.size shouldBe 0
        
        val json = objectMapper.writeValueAsString(transaction)
        val deserialized: Transaction = objectMapper.readValue(json)
        
        deserialized.features.size shouldBe 0
    }

    test("Transaction should support data class copy") {
        val original = Transaction(
            id = "txn-005",
            timestamp = 1705334800000L,
            amount = 100.0,
            merchantCategory = "retail",
            features = mapOf("V1" to 1.0)
        )

        val copied = original.copy(amount = 200.0)

        copied.id shouldBe original.id
        copied.timestamp shouldBe original.timestamp
        copied.amount shouldBe 200.0
        copied.merchantCategory shouldBe original.merchantCategory
        copied.features shouldBe original.features
    }
})
