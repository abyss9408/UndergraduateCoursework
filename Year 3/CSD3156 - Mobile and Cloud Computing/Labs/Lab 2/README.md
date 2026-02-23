# csd3156-lab02-2026

## Your task
- The question is in XSite -> Content -> Mobile Computing Labs
- fork and clone the repository
- commit and push the changes

## Test your solution
- The UIInstrumentedTest in androidTest is for testing the UI test tags for the app
- You may run the ./test.bat or ./test.sh in the test folder in the root directory of the project to run the Grading apk
    - The test result will be shown in the console
    - You need to set the Enviroment Variable for the test to run
    
        - Add adb to the PATH, i.e, add SDK_location/platform-tools/, SDK_location can be found in Settings->Language & Frameworks->Android SDK
        - Add JAVA_HOME to the PATH, just use the embedded Gradle JDK path in Android Studio, it can be found in Settings->Build, Execution, Deployment->Build Tools->Gradle
        - Ensure the Emulator or the device is connected and running
        - You may need to run multiple times