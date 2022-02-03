package org.tensorflow.lite.examples.poseestimation

class Rep {
    var id: Int = 0
    var time: Float = 0f
    var mean_time: Float = 0f
    var median_time: Float = 0f
    var is_last: Int = 0

    constructor(time: Float, mean_time: Float, median_time: Float, is_last: Int) {
        this.time = time
        this.mean_time = mean_time
        this.median_time = median_time
        this.is_last = is_last
    }
    constructor(){}
}