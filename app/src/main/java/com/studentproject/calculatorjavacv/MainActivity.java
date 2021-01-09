package com.studentproject.calculatorjavacv;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2
{

    JavaCameraView javaCameraView;
    Mat mRGBA, mRGBAT;
    private Interpreter tflite;
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final String MODEL_PATH = "mnist.tflite";

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(MainActivity.this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case BaseLoaderCallback.SUCCESS:
                {
                    //javaCameraView.enableView();
                    break;
                }
                default:
                {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    private MappedByteBuffer loadModelFile() throws IOException
    {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    //checks if app can use camera
    public void checkPermission(String permission, int requestCode)
    {
        // Checking if permission is not granted
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[] { permission }, requestCode);
        }
    }
    //ask for camera permission if needed
    public void onRequestPermissionsResult(int requestCode,@NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(MainActivity.this,
                        "Camera Permission Granted",
                        Toast.LENGTH_SHORT)
                        .show();
            }
            else {
                Toast.makeText(MainActivity.this,
                        "Camera Permission Denied",
                        Toast.LENGTH_SHORT)
                        .show();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        checkPermission(Manifest.permission.CAMERA, CAMERA_PERMISSION_CODE);

        try
        {
            tflite = new Interpreter(loadModelFile());
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height)
    {
        mRGBA = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped()
    {
        mRGBA.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        mRGBA = inputFrame.rgba();
        ConvertFrame cFrame = new ConvertFrame();
        //draw rectangle
        Rect targetFrame = new Rect((int)(mRGBA.width()*0.05), 100, (int)(mRGBA.width()*0.9), 200);
        Imgproc.rectangle(mRGBA, targetFrame.tl(), targetFrame.br(), new Scalar(0, 0, 205), 3);
        Mat mCROP = new Mat(mRGBA.clone(), targetFrame);
        Mat mTHRESH = new Mat();
//        mTHRESH to list of contours
        mTHRESH = new ConvertFrame().BlackWhiteFrame(mCROP);
//        Mat mTemp = mRGBA.clone();
//        mRGBA = new ConvertFrame().BcgFrameMerger(mTHRESH, mTemp);
        List<MatOfPoint> mEDGES = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mTHRESH, mEDGES, hierarchy,Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE); //RETR_EXTERNAL
        if (!mEDGES.isEmpty() && !hierarchy.empty())
        {
            List<MatOfPoint> mEDGES_H = new ArrayList<MatOfPoint>();
            for(int i = 0; i < mEDGES.size(); i++)
            {
                mEDGES_H.add(mEDGES.get(i));
                if (hierarchy.get(0, i)[1] == -1)
                {
                    mEDGES_H.add(mEDGES.get(i));
                }
            }
            Mat mSKIN = markContoursOnFrame(mEDGES_H, mRGBA);
            mRGBAT = mSKIN.t();
            Core.flip(mSKIN,mRGBAT,1);
            Imgproc.resize(mRGBAT,mRGBAT,mRGBA.size()); // przeskalowanie klatki
            return mRGBAT; //zwrócenie klatki do wyświetlenia
        }
        else
        {
            return mRGBA;
        }
    }

    private Mat markContoursOnFrame(List<MatOfPoint> listOfContours, Mat zerosFrame)
    {
        Mat frame = zerosFrame.clone();
        Point offset = new Point((int)(frame.width()*0.05), 100);
        MatOfPoint2f figures;
        List<Rect> boundRects = new ArrayList<Rect>();
        String tt = "";

        Rect boundRect ;//= new Rect();
        for (int i = 0; i < listOfContours.size(); i++) {
            figures = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(listOfContours.get(i).toArray()), figures, 3, true);
            boundRect = Imgproc.boundingRect(new MatOfPoint(figures.toArray()));
            try
            {
                if ((Math.abs(boundRect.tl().x - boundRect.br().x) > 20) && (Math.abs(boundRect.tl().x - boundRect.br().x) < (frame.width()/2)) && (Math.abs(boundRect.tl().y - boundRect.br().y)) > 20 && (Math.abs(boundRect.tl().y - boundRect.br().y)) < frame.height())
                {
                    boundRects.add(boundRect);
                }
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
        }

        int foundTLX = 0;
        int foundTLY = 0;
        int foundBRX = 0;
        int foundBRY = 0;

        try
        {
            for (int i = 1; i < boundRects.size(); i++)
            {
                if (boundRects.get(i).tl().x < boundRects.get(foundTLX).tl().x) foundTLX = i;
                if (boundRects.get(i).tl().y < boundRects.get(foundTLY).tl().y) foundTLY = i;
                if (boundRects.get(i).br().x > boundRects.get(foundBRX).br().x) foundBRX = i;
                if (boundRects.get(i).br().y > boundRects.get(foundBRY).br().y) foundBRY = i;
            }
        }
        catch (Exception exc)
        {
            exc.printStackTrace();
        }

        Point LR;
        Point UL;
        try
        {
            UL = new Point(boundRects.get(foundTLX).tl().x + (int)(frame.width()*0.05), boundRects.get(foundTLY).tl().y + 100);
            LR = new Point(boundRects.get(foundBRX).br().x + (int)(frame.width()*0.05), boundRects.get(foundBRY).br().y + 100);
        }
        catch(Exception exct)
        {
            UL = new Point((int)(frame.width()*0.05),100);
            LR = new Point((int)(frame.width()*0.05),100);
        }
        Imgproc.rectangle(frame, UL, LR, new Scalar(0, 255, 0), 3);
        for (Rect rect: boundRects)
        {
            Point TL = new Point(rect.tl().x + offset.x, rect.tl().y + offset.y);
            Point BR = new Point(rect.br().x + offset.x, rect.br().y + offset.y);
            Imgproc.rectangle(frame, TL, BR, new Scalar(255, 255, 0), 3);
        }
        return frame;
    }

    private String classify(Mat mat)
    {
        Mat mHSV = new Mat(mat.height(), mat.width(), CvType.CV_32F);
        mat.convertTo(mHSV, CvType.CV_32F);
        float [] buffer = new float[(int) (mHSV.total()*mHSV.channels())];
        mHSV.get(0,0, buffer);
        float[][] output = new float[1][10];
        tflite.run(buffer, output);
        return String.valueOf(output[0]);
    }

    private String classify (List<Rect> boundRects, Mat frame)
    {
        String text = "";
        Point offset = new Point((int)(frame.width()*0.05), 100);
        List<float[]> buffers = new ArrayList<>();
        Map<Integer, Object> outputs = new HashMap<>();
        for (Rect rect : boundRects)
        {
            Point TL = new Point(rect.tl().x + offset.x, rect.tl().y + offset.y);
            Point BR = new Point(rect.br().x + offset.x, rect.br().y + offset.y);
            Imgproc.rectangle(frame.clone(), TL, BR, new Scalar(255, 255, 0), 3);
            Mat model = new Mat(frame, new Rect((int)TL.x, (int)BR.y, (int)Math.abs(TL.x - BR.x), (int)Math.abs(TL.y - BR.y)));
            Mat mHSV = new Mat(model.height(), model.width(), CvType.CV_32F);
            model.convertTo(mHSV, CvType.CV_32F);
            float [] buffer = new float[(int) (mHSV.total()*mHSV.channels())];
            mHSV.get(0,0, buffer);
            buffers.add(buffer);
        }
        tflite.runForMultipleInputsOutputs(buffers.toArray() ,outputs);
        for (Map.Entry<Integer, Object> output : outputs.entrySet())
        {
            text += output.getValue().toString();
        }
        return text;
    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();

        if (javaCameraView != null)
        {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        if (javaCameraView != null)
        {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume()
    {
        super.onResume();

        if (OpenCVLoader.initDebug())
        {
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
        else
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }
}