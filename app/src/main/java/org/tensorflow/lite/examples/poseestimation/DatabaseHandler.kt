package org.tensorflow.lite.examples.poseestimation

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.widget.Toast

val DATABASE_NAME = "FitnessDB"
val TABLE_NAME = "Rep"
var COL_TIME = "time"
var COL_MEAN_TIME = "mean_time"
var COL_MEDIAN_TIME = "median_time"
var COL_IS_LAST = "is_last"
var COL_ID = "id"

class DatabaseHandler(var context: Context): SQLiteOpenHelper(context, DATABASE_NAME, null, 1){
    override fun onCreate(db: SQLiteDatabase?) {

        val createTable = "CREATE TABLE ${TABLE_NAME} (" +
                "${COL_ID} INTEGER PRIMARY KEY AUTOINCREMENT," +
                "${COL_TIME} FLOAT," +
                "${COL_MEAN_TIME} FLOAT," +
                "${COL_MEDIAN_TIME} FLOAT," +
                "${COL_IS_LAST} INTEGER)"
        db?.execSQL(createTable)
    }

    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {
        TODO("Not yet implemented")
    }
    fun insertData(rep:Rep){
        val db = this.writableDatabase
        var cv = ContentValues()
        cv.put(COL_TIME, rep.time)
        cv.put(COL_MEAN_TIME, rep.mean_time)
        cv.put(COL_MEDIAN_TIME, rep.median_time)
        cv.put(COL_IS_LAST, rep.is_last)
        db.insert(TABLE_NAME, null, cv)
    }

    fun readData():MutableList<Rep>{
        var list: MutableList<Rep> = ArrayList()

        val db = this.readableDatabase
        val query = "Select * from " + TABLE_NAME
        val result = db.rawQuery(query, null)
        if (result.moveToFirst()){
            do{
                var rep = Rep()
                rep.id = result.getString(result.getColumnIndex(COL_ID)).toInt()
                rep.time = result.getString(result.getColumnIndex(COL_TIME)).toFloat()
                rep.mean_time = result.getString(result.getColumnIndex(COL_MEAN_TIME)).toFloat()
                rep.median_time = result.getString(result.getColumnIndex(COL_MEDIAN_TIME)).toFloat()
                rep.is_last = result.getString(result.getColumnIndex(COL_IS_LAST)).toInt()
                list.add(rep)
            }while (result.moveToNext())
        }

        result.close()
        db.close()

        return  list
    }

    fun deleteData(){
        val db = this.writableDatabase
        db.execSQL("delete from "+ TABLE_NAME)
        db.close()
    }

}