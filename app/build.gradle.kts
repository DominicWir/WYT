import org.gradle.kotlin.dsl.invoke

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.wytv2"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.example.wytv2"
        minSdk = 24
        targetSdk = 33  // Lower to avoid 16KB enforcement
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        create("release") {
            // Values injected by GitHub Actions secrets (see .github/workflows/release.yml)
            // Leave unset for local debug builds — release signing only runs in CI.
            val keystoreFile = System.getenv("KEYSTORE_FILE")
            if (keystoreFile != null) {
                storeFile = file(keystoreFile)
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                keyAlias = System.getenv("KEY_ALIAS")
                keyPassword = System.getenv("KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            val keystoreFile = System.getenv("KEYSTORE_FILE")
            if (keystoreFile != null) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
    
    aaptOptions {
        noCompress("onnx", "data")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    // ONNX Runtime for ML model inference
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.0")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}