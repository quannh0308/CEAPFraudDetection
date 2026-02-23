plugins {
    kotlin("jvm")
    id("com.github.johnrengelman.shadow")
}

dependencies {
    // Internal dependencies
    implementation(project(":fraud-detection-common"))
    
    // CEAP platform dependencies
    implementation(files("../ceap-platform/ceap-workflow-etl/build/libs/ceap-workflow-etl-1.0.0-SNAPSHOT.jar"))
    implementation(files("../ceap-platform/ceap-common/build/libs/ceap-common-1.0.0-SNAPSHOT.jar"))
    implementation(files("../ceap-platform/ceap-models/build/libs/ceap-models-1.0.0-SNAPSHOT.jar"))
    
    // Jackson for JSON serialization
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.0")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.16.0")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.0")
    
    // AWS SDK v2
    implementation(platform("software.amazon.awssdk:bom:2.21.0"))
    implementation("software.amazon.awssdk:s3")
    implementation("software.amazon.awssdk:sagemaker")
    implementation("software.amazon.awssdk:sagemakerruntime")
    implementation("software.amazon.awssdk:ssm")
    
    // AWS Lambda
    implementation("com.amazonaws:aws-lambda-java-core:1.2.3")
    implementation("com.amazonaws:aws-lambda-java-events:3.11.3")
    
}

tasks.shadowJar {
    archiveBaseName.set("fraud-training-pipeline")
    archiveClassifier.set("")
    archiveVersion.set("")
    
    // Merge service files for AWS SDK
    mergeServiceFiles()
    
    // Exclude signature files
    exclude("META-INF/*.SF")
    exclude("META-INF/*.DSA")
    exclude("META-INF/*.RSA")
}

tasks.build {
    dependsOn(tasks.shadowJar)
}
