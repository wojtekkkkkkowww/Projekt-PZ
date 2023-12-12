package com.example.aplikacja_do_gestow
import android.content.Context
import com.example.aplikacja_do_gestow.ml.Model1
import com.example.aplikacja_do_gestow.ml.Model2
import com.example.aplikacja_do_gestow.ml.Model3
import com.example.aplikacja_do_gestow.ml.Supermodel
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Models(context: Context) {
    private val model1 = Model1.newInstance(context)
    private val model2 = Model2.newInstance(context)
    private val model3 = Model3.newInstance(context)
    private val supermodel = Supermodel.newInstance(context)

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
        return byteBuffer
    }

    private fun resultsToByteBuffer(result1:FloatArray,result2: FloatArray,result3:FloatArray):ByteBuffer{
        val byteBuffer = ByteBuffer.allocateDirect(4*1*3*12)
        byteBuffer.order(ByteOrder.nativeOrder())

        for(value in result1) {
                byteBuffer.putFloat(value)
        }
        for(value in result2) {
            byteBuffer.putFloat(value)
        }
        for(value in result3) {
            byteBuffer.putFloat(value)
        }

        return byteBuffer
    }


    private fun softmax(inputArray: FloatArray): FloatArray {
        val max = inputArray.maxOrNull() ?: 0f
        val expArray = inputArray.map { kotlin.math.exp(it - max) }.toFloatArray()
        val sumExp = expArray.sum()
        return expArray.map { it / sumExp }.toFloatArray()
    }
    fun predict(handLandmarkerResult: HandLandmarkerResult): FloatArray {
        val byteBuffer = landmarkToByteBuffer(handLandmarkerResult)
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 21, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        val output1 = model1.process(inputFeature0)
        val output2 = model2.process(inputFeature0)
        val output3 = model3.process(inputFeature0)

        val outputFeature1 = output1.outputFeature0AsTensorBuffer
        val outputFeature2 = output2.outputFeature0AsTensorBuffer
        val outputFeature3 = output3.outputFeature0AsTensorBuffer

        val resultArray1 = softmax(outputFeature1.floatArray)
        val resultArray2 = softmax(outputFeature2.floatArray)
        val resultArray3 = softmax(outputFeature3.floatArray)


        val superByteBuffer = resultsToByteBuffer(resultArray1, resultArray2, resultArray3)

        val superInputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 3, 12), DataType.FLOAT32)

        superInputFeature.loadBuffer(superByteBuffer)


        val superOutput = supermodel.process(superInputFeature)
        val superOutputFeature = superOutput.outputFeature0AsTensorBuffer

        return softmax(superOutputFeature.floatArray)

    }


}