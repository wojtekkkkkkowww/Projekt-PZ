package com.example.aplikacja_do_gestow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.mediapipe.framework.image.BitmapImageBuilder
import androidx.camera.core.ImageProxy
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult


class HandLandmarkerProcessor(
    val context: Context
) {
    var isFrontCamera = true
    private var handLandmarker: HandLandmarker? = null

    init {
        setupHandLandmarker()
    }

    private fun setupHandLandmarker() {
        // Initialize handLandmarker based on your requirements
        // (e.g., set confidence thresholds, delegate, etc.)
        val baseOptionsBuilder = BaseOptions.builder()
        baseOptionsBuilder.setModelAssetPath("hand_landmarker.task")
        baseOptionsBuilder.setDelegate(Delegate.GPU)

        val baseOptions = baseOptionsBuilder.build()

        val optionsBuilder =
            HandLandmarker.HandLandmarkerOptions.builder().setBaseOptions(baseOptions)
                .setMinHandDetectionConfidence(0.5f).setMinTrackingConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f).setNumHands(1).setRunningMode(RunningMode.IMAGE)

        val options = optionsBuilder.build()
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun processImage(imageProxy: ImageProxy): List<HandLandmarkerResult> {


        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
        )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
        )
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()
        val detectionResult = handLandmarker?.detect(mpImage)
        return if (detectionResult != null) listOf(detectionResult)
        else listOf()
    }
}