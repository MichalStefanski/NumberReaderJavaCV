package com.studentproject.calculatorjavacv;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Classifier
{
    private static final String MODEL_PATH = "mnist.tflite";
    private float[][] output = null;
    protected ByteBuffer imgBuffer = null;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE =1;
    private static final int  DIM_HEIGHT =28;
    private static final int DIM_WIDTH = 28;
    private static final int BYTES = 4;

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException
    {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Interpreter tflite;

    Classifier(Activity activity) throws IOException
    {
        tflite = new Interpreter(loadModelFile(activity));
        imgBuffer = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_PIXEL_SIZE * BYTES);
        imgBuffer.order(ByteOrder.nativeOrder());
        output = new float[1][10];
    }

    private void classify(Mat mat)
    {
        if(tflite!=null)
        {
            imgBuffer.rewind();
            int px = 0;
            for (int i = 0; i < DIM_HEIGHT; i++)
            {
                for (int j = 0; j < DIM_WIDTH; j++)
                {
                    imgBuffer.putFloat((float)mat.get(i,j)[0]);
                }
            }

            if (imgBuffer != null)
            {
                tflite.run(imgBuffer, output);
            }
        }
    }

    public String getResult(Mat mat)
    {
        String result = "";
        int index = -1;
        float score = 0f;
        classify(new ConvertFrame().ScaleToModel(mat));

        for (int i = 0; i < output[0].length; i++)
        {
            if (output[0][i] > score)
            {
                score = output[0][i];
                index = i;
            }
        }
        if (index == -1) return "";
        return String.valueOf(index);
    }
}
