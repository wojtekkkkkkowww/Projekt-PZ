package com.example.aplikacja_do_gestow

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import com.example.aplikacja_do_gestow.databinding.ActivityMainBinding
import com.example.aplikacja_do_gestow.ml.Model1
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() ,HandLandmarkerProcessor.LandmarkerListener {
    lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    var prediction = 0

    private lateinit var cameraExecutor: ExecutorService

    lateinit var handLandmarkerProcessor: HandLandmarkerProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }
        viewBinding.textView.text = prediction.toString()
        handLandmarkerProcessor = HandLandmarkerProcessor(this,this)
        cameraExecutor = Executors.newSingleThreadExecutor()

    }


    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(viewBinding.viewFinder.display.rotation)
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(viewBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor){
                        image->detectHand(image)
                    }
                }

            // Select front camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA,
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }


    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        )
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(
                    baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                startCamera()
            }
        }


    private fun landmarkToByteBuffer(handLandmarkerResult: HandLandmarkerResult): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4*21*3)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (landmark in handLandmarkerResult.worldLandmarks()) {
            for (L in landmark) {
                byteBuffer.putFloat(L.x())
                byteBuffer.putFloat(L.y())
                byteBuffer.putFloat(L.z())
            }
        }
        //byteBuffer.rewind()
        return byteBuffer
    }


    fun softmax(inputArray: FloatArray): FloatArray {
        val max = inputArray.maxOrNull() ?: 0f
        val expArray = inputArray.map { kotlin.math.exp(it - max) }.toFloatArray()
        val sumExp = expArray.sum()
        return expArray.map { it / sumExp }.toFloatArray()
    }


    private fun detectHand(imageProxy: ImageProxy){
        handLandmarkerProcessor.processImage(imageProxy)
    }

    override fun onError(error: String, errorCode: Int) {
        TODO("Not yet implemented")

    }


    override fun onResult(resultBundle: HandLandmarkerProcessor.ResultBundle) {
           viewBinding.overly.setResults(
            resultBundle.result.first(),
            resultBundle.inputImageHeight,
            resultBundle.inputImageWidth,
           )
            if (resultBundle.result.first().landmarks().isNotEmpty()) {

                //Test Model1
                val byteBuffer = landmarkToByteBuffer(resultBundle.result.first())
                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 21, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(byteBuffer)

                val model1 = Model1.newInstance(this)
                val outputs = model1.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                val resultArray = outputFeature0.floatArray
                val maxValue = resultArray.maxOrNull()
                prediction = resultArray.indexOfFirst { it == maxValue } + 1
                viewBinding.textView.text = prediction.toString()
                Log.d(TAG,"array:${resultArray.contentToString()}  " )
                Log.d(TAG,"arraysoft:${softmax(resultArray).contentToString()}  " )
                Log.d(TAG, "WYNIK: $prediction ")
                Log.d(TAG, "WORDLANDMARK ${resultBundle.result.first().worldLandmarks()} ")

            }else{
                prediction = 0
                viewBinding.textView.text = prediction.toString()
            }
    }

}