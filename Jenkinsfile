 node('bose') {
   stage('Checkout') {

        checkout scm 
    }
    try {
    stage('Run the NoseTests') {
          sh '''#!/bin/bash -l
          echo "######NoseTest #######"·
          docker-compose up
          '''
          junit 'results.xml'
      }
    } catch (e) {
      notifyFailed()
      throw e;
      }
}
