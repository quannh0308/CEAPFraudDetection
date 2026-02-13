plugins {
    kotlin("jvm")
    application
}

group = "com.frauddetection"
version = "1.0.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    // AWS CDK
    implementation("software.amazon.awscdk:aws-cdk-lib:2.114.1")
    implementation("software.constructs:constructs:10.3.0")
    
    // Kotlin
    implementation(kotlin("stdlib"))
    
    // Testing
    testImplementation("io.kotest:kotest-runner-junit5:5.8.0")
    testImplementation("io.kotest:kotest-assertions-core:5.8.0")
    testImplementation("io.kotest:kotest-property:5.8.0")
    testImplementation("software.amazon.awscdk:aws-cdk-lib:2.114.1")
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("com.frauddetection.infrastructure.FraudDetectionAppKt")
}
