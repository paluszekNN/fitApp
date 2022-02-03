/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.ml

import android.content.Context
import android.graphics.*
import android.media.MediaPlayer
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.DatabaseHandler
import org.tensorflow.lite.examples.poseestimation.data.*
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.Exception
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.tensorflow.lite.examples.poseestimation.R
import org.tensorflow.lite.examples.poseestimation.textCounter

enum class ModelType {
    Lightning,
    Thunders
}
var counter = 0
var start_rep = 0L
var reps_time = mutableListOf<Float>()
var reps_mean_wrist_pos = mutableListOf<Float>()
var reps_median_wrist_pos = mutableListOf<Float>()
var rep_wrist_poses_per_time = mutableListOf<Float>()

class MoveNet(private val interpreter: Interpreter, private var gpuDelegate: GpuDelegate?, context: Context) :
    PoseDetector {

    companion object {
        private const val MIN_CROP_KEYPOINT_SCORE = .2f
        private const val CPU_NUM_THREADS = 4

        // Parameters that control how large crop region should be expanded from previous frames'
        // body keypoints.
        private const val TORSO_EXPANSION_RATIO = 1.9f
        private const val BODY_EXPANSION_RATIO = 1.2f

        // TFLite file names.
        private const val LIGHTNING_FILENAME = "movenet_lightning.tflite"
        private const val THUNDER_FILENAME = "movenet_thunder.tflite"

        // allow specifying model type.
        fun create(context: Context, device: Device, modelType: ModelType): MoveNet {
            val options = Interpreter.Options()
            var gpuDelegate: GpuDelegate? = null
            options.setNumThreads(CPU_NUM_THREADS)
            when (device) {
                Device.CPU -> {
                }
                Device.GPU -> {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                }
                Device.NNAPI -> options.setUseNNAPI(true)
            }
            return MoveNet(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        if (modelType == ModelType.Lightning) LIGHTNING_FILENAME
                        else THUNDER_FILENAME
                    ), options
                ),
                gpuDelegate,
                context
            )
        }
        // default to lightning.
        fun create(context: Context, device: Device): MoveNet =
            create(context, device, ModelType.Lightning)
    }

    private var cropRegion: RectF? = null
    private var lastInferenceTimeNanos: Long = -1
    private val inputWidth = interpreter.getInputTensor(0).shape()[1]
    private val inputHeight = interpreter.getInputTensor(0).shape()[2]
    private var outputShape: IntArray = interpreter.getOutputTensor(0).shape()
    private var mp: MediaPlayer? = null
    private var mp2: MediaPlayer? = null
    private var up = true
    private var down = false
    private var context: Context? = context


    override fun estimatePoses(bitmap: Bitmap): List<Person> {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        if (cropRegion == null) {
            cropRegion = initRectF(bitmap.width, bitmap.height)
        }
        var totalScore = 0f

        val numKeyPoints = outputShape[2]
        val keyPoints = mutableListOf<KeyPoint>()

        cropRegion?.run {
            val rect = RectF(
                (left * bitmap.width),
                (top * bitmap.height),
                (right * bitmap.width),
                (bottom * bitmap.height)
            )
            val detectBitmap = Bitmap.createBitmap(
                rect.width().toInt(),
                rect.height().toInt(),
                Bitmap.Config.ARGB_8888
            )
            Canvas(detectBitmap).drawBitmap(
                bitmap,
                -rect.left,
                -rect.top,
                null
            )
            val inputTensor = processInputImage(detectBitmap, inputWidth, inputHeight)
            val outputTensor = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
            val widthRatio = detectBitmap.width.toFloat() / inputWidth
            val heightRatio = detectBitmap.height.toFloat() / inputHeight

            val positions = mutableListOf<Float>()

            inputTensor?.let { input ->
                interpreter.run(input.buffer, outputTensor.buffer.rewind())
                val output = outputTensor.floatArray
                for (idx in 0 until numKeyPoints) {
                    val x = output[idx * 3 + 1] * inputWidth * widthRatio
                    val y = output[idx * 3 + 0] * inputHeight * heightRatio

                    positions.add(x)
                    positions.add(y)
                    val score = output[idx * 3 + 2]
                    keyPoints.add(
                        KeyPoint(
                            BodyPart.fromInt(idx),
                            PointF(
                                x,
                                y
                            ),
                            score
                        )
                    )
                    totalScore += score
                }
            }
            val matrix = Matrix()
            val points = positions.toFloatArray()

            matrix.postTranslate(rect.left, rect.top)
            matrix.mapPoints(points)
            keyPoints.forEachIndexed { index, keyPoint ->
                keyPoint.coordinate =
                    PointF(
                        points[index * 2],
                        points[index * 2 + 1]
                    )
            }
            // new crop region
            cropRegion = determineRectF(keyPoints, bitmap.width, bitmap.height)
        }
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        var position_of_left_shoulder_x = 0.0
        var position_of_left_shoulder_y = 0.0
        var position_of_right_shoulder_x = 0.0
        var position_of_right_shoulder_y = 0.0
        var position_of_left_elbow_x = 0.0
        var position_of_left_elbow_y = 0.0
        var position_of_right_elbow_x = 0.0
        var position_of_right_elbow_y = 0.0
        var position_of_right_wrist_x = 0.0
        var position_of_right_wrist_y = 0.0
        var position_of_left_wrist_x = 0.0
        var position_of_left_wrist_y = 0.0
        for (keyPoint in keyPoints) {
            if (keyPoint.score > 0.5) {
                if (keyPoint.bodyPart == BodyPart.LEFT_ELBOW){
                    position_of_left_elbow_x = keyPoint.coordinate.x.toDouble()
                    position_of_left_elbow_y = keyPoint.coordinate.y.toDouble()
                }
                if (keyPoint.bodyPart == BodyPart.RIGHT_ELBOW){
                    position_of_right_elbow_x = keyPoint.coordinate.x.toDouble()
                    position_of_right_elbow_y = keyPoint.coordinate.y.toDouble()
                }
                if (keyPoint.bodyPart == BodyPart.LEFT_WRIST){
                    position_of_left_wrist_x = keyPoint.coordinate.x.toDouble()
                    position_of_left_wrist_y = keyPoint.coordinate.y.toDouble()
                }
                if (keyPoint.bodyPart == BodyPart.RIGHT_WRIST){
                    position_of_right_wrist_x = keyPoint.coordinate.x.toDouble()
                    position_of_right_wrist_y = keyPoint.coordinate.y.toDouble()
                }
                if (keyPoint.bodyPart == BodyPart.LEFT_SHOULDER){
                    position_of_left_shoulder_x = keyPoint.coordinate.x.toDouble()
                    position_of_left_shoulder_y = keyPoint.coordinate.y.toDouble()
                }
                if (keyPoint.bodyPart == BodyPart.RIGHT_SHOULDER){
                    position_of_right_shoulder_x = keyPoint.coordinate.x.toDouble()
                    position_of_right_shoulder_y = keyPoint.coordinate.y.toDouble()
                }

                if ((SystemClock.elapsedRealtimeNanos() - start_rep - (start_rep%100))/100 > 2L && down){
                    rep_wrist_poses_per_time.add((((position_of_right_wrist_x - position_of_right_elbow_x)*(position_of_right_shoulder_x - position_of_right_elbow_x)+
                            (position_of_right_wrist_y - position_of_right_elbow_y)*(position_of_right_shoulder_y - position_of_right_elbow_y))/
                            (sqrt(Math.pow(position_of_right_wrist_x - position_of_right_elbow_x,2.0) +
                                    Math.pow(position_of_right_wrist_y - position_of_right_elbow_y,2.0)) *
                                    sqrt(Math.pow(position_of_right_elbow_x - position_of_right_shoulder_x,2.0) +
                                            Math.pow(position_of_right_elbow_y - position_of_right_shoulder_y,2.0)))).toFloat()
                    )
                }
//                var db = this.context?.let { DatabaseHandler(it) }
//                var data = db!!.readData()
//                var logText = ""
//                for (i in 0..data.size-1){
//                    logText += (data.get(i).time.toString() + " " + data.get(i).mean_time.toString() + " " +data.get(i).median_time.toString() + " " +data.get(i).is_last.toString()+"\n")
//                }
//                Log.d("database",logText)
//                if (position_of_left_shoulder_x !=0.0 &&
//                    position_of_left_shoulder_y !=0.0 &&
//                    position_of_right_shoulder_x !=0.0 &&
//                    position_of_right_shoulder_y !=0.0 &&
//                    position_of_left_elbow_x !=0.0 &&
//                    position_of_left_elbow_y !=0.0 &&
//                    position_of_right_elbow_x !=0.0 &&
//                    position_of_right_elbow_y !=0.0 &&
//                    position_of_right_wrist_x !=0.0 &&
//                    position_of_right_wrist_y !=0.0 &&
//                    position_of_left_wrist_x !=0.0 &&
//                    position_of_left_wrist_y !=0.0) {
//                    Log.d(
//                        "right", (((position_of_right_wrist_x - position_of_right_elbow_x)*(position_of_right_shoulder_x - position_of_right_elbow_x)+
//                                (position_of_right_wrist_y - position_of_right_elbow_y)*(position_of_right_shoulder_y - position_of_right_elbow_y))/
//                                (sqrt(Math.pow(position_of_right_wrist_x - position_of_right_elbow_x,2.0) +
//                                        Math.pow(position_of_right_wrist_y - position_of_right_elbow_y,2.0)) *
//                                        sqrt(Math.pow(position_of_right_elbow_x - position_of_right_shoulder_x,2.0) +
//                                                Math.pow(position_of_right_elbow_y - position_of_right_shoulder_y,2.0)))).toString()
//                    )
//                    Log.d("left", (((position_of_left_wrist_x - position_of_left_elbow_x)*(position_of_left_shoulder_x - position_of_left_elbow_x)+
//                            (position_of_left_wrist_y - position_of_left_elbow_y)*(position_of_left_shoulder_y - position_of_left_elbow_y))/
//                            (sqrt(Math.pow(position_of_left_wrist_x - position_of_left_elbow_x,2.0) +
//                                    Math.pow(position_of_left_wrist_y - position_of_left_elbow_y,2.0)) *
//                                    sqrt(Math.pow(position_of_left_elbow_x - position_of_left_shoulder_x,2.0) +
//                                            Math.pow(position_of_left_elbow_y - position_of_left_shoulder_y,2.0)))).toString()
//                    )
//                }
                if (position_of_left_shoulder_x !=0.0 &&
                    position_of_left_shoulder_y !=0.0 &&
                    position_of_right_shoulder_x !=0.0 &&
                    position_of_right_shoulder_y !=0.0 &&
                    position_of_left_elbow_x !=0.0 &&
                    position_of_left_elbow_y !=0.0 &&
                    position_of_right_elbow_x !=0.0 &&
                    position_of_right_elbow_y !=0.0 &&
                    position_of_right_wrist_x !=0.0 &&
                    position_of_right_wrist_y !=0.0 &&
                    position_of_left_wrist_x !=0.0 &&
                    position_of_left_wrist_y !=0.0 && !up && (((position_of_right_wrist_x - position_of_right_elbow_x)*(position_of_right_shoulder_x - position_of_right_elbow_x)+
                            (position_of_right_wrist_y - position_of_right_elbow_y)*(position_of_right_shoulder_y - position_of_right_elbow_y))/
                    (sqrt(Math.pow(position_of_right_wrist_x - position_of_right_elbow_x,2.0) +
                            Math.pow(position_of_right_wrist_y - position_of_right_elbow_y,2.0)) *
                            sqrt(Math.pow(position_of_right_elbow_x - position_of_right_shoulder_x,2.0) +
                                    Math.pow(position_of_right_elbow_y - position_of_right_shoulder_y,2.0)))) < -0.85 && (((position_of_left_wrist_x - position_of_left_elbow_x)*(position_of_left_shoulder_x - position_of_left_elbow_x)+
                            (position_of_left_wrist_y - position_of_left_elbow_y)*(position_of_left_shoulder_y - position_of_left_elbow_y))/
                            (sqrt(Math.pow(position_of_left_wrist_x - position_of_left_elbow_x,2.0) +
                                    Math.pow(position_of_left_wrist_y - position_of_left_elbow_y,2.0)) *
                                    sqrt(Math.pow(position_of_left_elbow_x - position_of_left_shoulder_x,2.0) +
                                            Math.pow(position_of_left_elbow_y - position_of_left_shoulder_y,2.0)))) < -0.85
                ) {
                    reps_time.add((SystemClock.elapsedRealtimeNanos() - start_rep).toFloat()/1000)
                    reps_mean_wrist_pos.add(rep_wrist_poses_per_time.average().toFloat())
                    if (rep_wrist_poses_per_time.size % 2 == 0)
                        reps_median_wrist_pos.add((rep_wrist_poses_per_time[rep_wrist_poses_per_time.size/2] + rep_wrist_poses_per_time[rep_wrist_poses_per_time.size/2 - 1])/2)
                    else
                        reps_median_wrist_pos.add(rep_wrist_poses_per_time[rep_wrist_poses_per_time.size/2])
                    down = false
                    up = true
                    counter += 1
                    textCounter.setText(counter.toString())
                    try {

                        if (mp == null)
                        {
                            mp = MediaPlayer.create(this.context, R.raw.alarm)
                            mp!!.start()
                        }
                        if (!mp!!.isPlaying){
                            mp!!.start()
                        }
                    } catch (e: Exception) {

                        e.printStackTrace()
                    }

                }
                if (position_of_left_shoulder_x !=0.0 &&
                    position_of_left_shoulder_y !=0.0 &&
                    position_of_right_shoulder_x !=0.0 &&
                    position_of_right_shoulder_y !=0.0 &&
                    position_of_left_elbow_x !=0.0 &&
                    position_of_left_elbow_y !=0.0 &&
                    position_of_right_elbow_x !=0.0 &&
                    position_of_right_elbow_y !=0.0 &&
                    position_of_right_wrist_x !=0.0 &&
                    position_of_right_wrist_y !=0.0 &&
                    position_of_left_wrist_x !=0.0 &&
                    position_of_left_wrist_y !=0.0 && !down &&
                    (((position_of_right_wrist_x - position_of_right_elbow_x)*(position_of_right_shoulder_x - position_of_right_elbow_x)+
                            (position_of_right_wrist_y - position_of_right_elbow_y)*(position_of_right_shoulder_y - position_of_right_elbow_y))/
                            (sqrt(Math.pow(position_of_right_wrist_x - position_of_right_elbow_x,2.0) +
                                    Math.pow(position_of_right_wrist_y - position_of_right_elbow_y,2.0)) *
                                    sqrt(Math.pow(position_of_right_elbow_x - position_of_right_shoulder_x,2.0) +
                                            Math.pow(position_of_right_elbow_y - position_of_right_shoulder_y,2.0)))) > 0.4 && (((position_of_left_wrist_x - position_of_left_elbow_x)*(position_of_left_shoulder_x - position_of_left_elbow_x)+
                            (position_of_left_wrist_y - position_of_left_elbow_y)*(position_of_left_shoulder_y - position_of_left_elbow_y))/
                            (sqrt(Math.pow(position_of_left_wrist_x - position_of_left_elbow_x,2.0) +
                                    Math.pow(position_of_left_wrist_y - position_of_left_elbow_y,2.0)) *
                                    sqrt(Math.pow(position_of_left_elbow_x - position_of_left_shoulder_x,2.0) +
                                            Math.pow(position_of_left_elbow_y - position_of_left_shoulder_y,2.0)))) > 0.4) {
                    start_rep = SystemClock.elapsedRealtimeNanos()
                    down = true
                    up = false
                    try {

                        if (mp2 == null)
                        {
                            mp2 = MediaPlayer.create(this.context, R.raw.alarm)
                            mp2!!.start()
                        }
                        if (!mp2!!.isPlaying){
                            mp2!!.start()
                        }
                    } catch (e: Exception) {

                        e.printStackTrace()
                    }

                }
            }
        }
        return listOf(Person(keyPoints = keyPoints, score = totalScore / numKeyPoints))
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos

    override fun close() {
        gpuDelegate?.close()
        interpreter.close()
        cropRegion = null
    }

    /**
     * Prepare input image for detection
     */
    private fun processInputImage(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): TensorImage? {
        val width: Int = bitmap.width
        val height: Int = bitmap.height

        val size = if (height > width) width else height
        val imageProcessor = ImageProcessor.Builder().apply {
            add(ResizeWithCropOrPadOp(size, size))
            add(ResizeOp(inputWidth, inputHeight, ResizeOp.ResizeMethod.BILINEAR))
        }.build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
    }

    /**
     * Defines the default crop region.
     * The function provides the initial crop region (pads the full image from both
     * sides to make it a square image) when the algorithm cannot reliably determine
     * the crop region from the previous frame.
     */
    private fun initRectF(imageWidth: Int, imageHeight: Int): RectF {
        val xMin: Float
        val yMin: Float
        val width: Float
        val height: Float
        if (imageWidth > imageHeight) {
            width = 1f
            height = imageWidth.toFloat() / imageHeight
            xMin = 0f
            yMin = (imageHeight / 2f - imageWidth / 2f) / imageHeight
        } else {
            height = 1f
            width = imageHeight.toFloat() / imageWidth
            yMin = 0f
            xMin = (imageWidth / 2f - imageHeight / 2) / imageWidth
        }
        return RectF(
            xMin,
            yMin,
            xMin + width,
            yMin + height
        )
    }

    /**
     * Checks whether there are enough torso keypoints.
     * This function checks whether the model is confident at predicting one of the
     * shoulders/hips which is required to determine a good crop region.
     */
    private fun torsoVisible(keyPoints: List<KeyPoint>): Boolean {
        return ((keyPoints[BodyPart.LEFT_HIP.position].score > MIN_CROP_KEYPOINT_SCORE).or(
            keyPoints[BodyPart.RIGHT_HIP.position].score > MIN_CROP_KEYPOINT_SCORE
        )).and(
            (keyPoints[BodyPart.LEFT_SHOULDER.position].score > MIN_CROP_KEYPOINT_SCORE).or(
                keyPoints[BodyPart.RIGHT_SHOULDER.position].score > MIN_CROP_KEYPOINT_SCORE
            )
        )
    }

    /**
     * Determines the region to crop the image for the model to run inference on.
     * The algorithm uses the detected joints from the previous frame to estimate
     * the square region that encloses the full body of the target person and
     * centers at the midpoint of two hip joints. The crop size is determined by
     * the distances between each joints and the center point.
     * When the model is not confident with the four torso joint predictions, the
     * function returns a default crop which is the full image padded to square.
     */
    private fun determineRectF(
        keyPoints: List<KeyPoint>,
        imageWidth: Int,
        imageHeight: Int
    ): RectF {
        val targetKeyPoints = mutableListOf<KeyPoint>()
        keyPoints.forEach {
            targetKeyPoints.add(
                KeyPoint(
                    it.bodyPart,
                    PointF(
                        it.coordinate.x,
                        it.coordinate.y
                    ),
                    it.score
                )
            )
        }
        if (torsoVisible(keyPoints)) {
            val centerX =
                (targetKeyPoints[BodyPart.LEFT_HIP.position].coordinate.x +
                        targetKeyPoints[BodyPart.RIGHT_HIP.position].coordinate.x) / 2f
            val centerY =
                (targetKeyPoints[BodyPart.LEFT_HIP.position].coordinate.y +
                        targetKeyPoints[BodyPart.RIGHT_HIP.position].coordinate.y) / 2f

            val torsoAndBodyDistances =
                determineTorsoAndBodyDistances(keyPoints, targetKeyPoints, centerX, centerY)

            val list = listOf(
                torsoAndBodyDistances.maxTorsoXDistance * TORSO_EXPANSION_RATIO,
                torsoAndBodyDistances.maxTorsoYDistance * TORSO_EXPANSION_RATIO,
                torsoAndBodyDistances.maxBodyXDistance * BODY_EXPANSION_RATIO,
                torsoAndBodyDistances.maxBodyYDistance * BODY_EXPANSION_RATIO
            )

            var cropLengthHalf = list.maxOrNull() ?: 0f
            val tmp = listOf(centerX, imageWidth - centerX, centerY, imageHeight - centerY)
            cropLengthHalf = min(cropLengthHalf, tmp.maxOrNull() ?: 0f)
            val cropCorner = Pair(centerY - cropLengthHalf, centerX - cropLengthHalf)

            return if (cropLengthHalf > max(imageWidth, imageHeight) / 2f) {
                initRectF(imageWidth, imageHeight)
            } else {
                val cropLength = cropLengthHalf * 2
                RectF(
                    cropCorner.second / imageWidth,
                    cropCorner.first / imageHeight,
                    (cropCorner.second + cropLength) / imageWidth,
                    (cropCorner.first + cropLength) / imageHeight,
                )
            }
        } else {
            return initRectF(imageWidth, imageHeight)
        }
    }

    /**
     * Calculates the maximum distance from each keypoints to the center location.
     * The function returns the maximum distances from the two sets of keypoints:
     * full 17 keypoints and 4 torso keypoints. The returned information will be
     * used to determine the crop size. See determineRectF for more detail.
     */
    private fun determineTorsoAndBodyDistances(
        keyPoints: List<KeyPoint>,
        targetKeyPoints: List<KeyPoint>,
        centerX: Float,
        centerY: Float
    ): TorsoAndBodyDistance {
        val torsoJoints = listOf(
            BodyPart.LEFT_SHOULDER.position,
            BodyPart.RIGHT_SHOULDER.position,
            BodyPart.LEFT_HIP.position,
            BodyPart.RIGHT_HIP.position
        )

        var maxTorsoYRange = 0f
        var maxTorsoXRange = 0f
        torsoJoints.forEach { joint ->
            val distY = abs(centerY - targetKeyPoints[joint].coordinate.y)
            val distX = abs(centerX - targetKeyPoints[joint].coordinate.x)
            if (distY > maxTorsoYRange) maxTorsoYRange = distY
            if (distX > maxTorsoXRange) maxTorsoXRange = distX
        }

        var maxBodyYRange = 0f
        var maxBodyXRange = 0f
        for (joint in keyPoints.indices) {
            if (keyPoints[joint].score < MIN_CROP_KEYPOINT_SCORE) continue
            val distY = abs(centerY - keyPoints[joint].coordinate.y)
            val distX = abs(centerX - keyPoints[joint].coordinate.x)

            if (distY > maxBodyYRange) maxBodyYRange = distY
            if (distX > maxBodyXRange) maxBodyXRange = distX
        }
        return TorsoAndBodyDistance(
            maxTorsoYRange,
            maxTorsoXRange,
            maxBodyYRange,
            maxBodyXRange
        )
    }
}
