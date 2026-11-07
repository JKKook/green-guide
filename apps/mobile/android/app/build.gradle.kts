import java.util.Properties

plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// Firebase(Crashlytics·Analytics) — google-services.json 은 사용자가 전달한다.
// 파일이 없으면 플러그인을 적용하지 않아 빌드가 깨지지 않고, 앱은 수집 없이 동작한다.
if (file("google-services.json").exists()) {
    apply(plugin = "com.google.gms.google-services")
    apply(plugin = "com.google.firebase.crashlytics")
}

// 릴리즈 서명 — android/key.properties 의 업로드 키로만 서명한다.
// key.properties 가 없으면 release 빌드를 실패시킨다(debug 키로 조용히 서명되는 것 방지).
// (key.properties / *.jks 는 VCS 제외. 내용 예시:
//   storeFile=upload-keystore.jks  storePassword=...  keyAlias=upload  keyPassword=...)
val keystoreProperties = Properties()
val keystorePropertiesFile = rootProject.file("key.properties")
val hasReleaseKeystore = keystorePropertiesFile.exists()
if (hasReleaseKeystore) {
    keystorePropertiesFile.inputStream().use { keystoreProperties.load(it) }
}

android {
    namespace = "com.greenguide.app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.greenguide.app"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    signingConfigs {
        if (hasReleaseKeystore) {
            create("release") {
                storeFile = rootProject.file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
            }
        }
    }

    buildTypes {
        release {
            if (hasReleaseKeystore) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }
}

// Kotlin 2.4 — kotlinOptions.jvmTarget(String) 제거, compilerOptions DSL 로 이관
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

flutter {
    source = "../.."
}

// release 태스크가 실행 그래프에 있는데 key.properties 가 없으면 즉시 실패 — debug·test 빌드는 영향 없음.
gradle.taskGraph.whenReady {
    if (!hasReleaseKeystore && allTasks.any { it.name.contains("Release") }) {
        throw GradleException(
            "android/key.properties 가 없어 release 빌드를 중단합니다 — 업로드 키 없이 서명할 수 없습니다."
        )
    }
}
