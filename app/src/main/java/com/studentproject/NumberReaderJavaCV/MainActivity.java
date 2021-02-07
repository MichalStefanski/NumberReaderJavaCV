package com.studentproject.NumberReaderJavaCV;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.view.View;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2
{

    JavaCameraView javaCameraView;
    Mat mRGBA, mRGBAT, mCROP, mTHRESH, mSKIN, mT, hierarchy;
    Mat kernel;
    TextView textView;
    Button calcButton;
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final String MODEL_PATH = "mnist.tflite";
    Classifier classifier;
    List <Mat> detectedContours;
    Mat t, v;
    List<Mat> tempList;

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(MainActivity.this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            if (status == BaseLoaderCallback.SUCCESS) {
                javaCameraView.enableView();
            } else {
                super.onManagerConnected(status);
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

        javaCameraView = findViewById(R.id.basic_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(MainActivity.this);

        textView = findViewById(R.id.predicted_text);
        calcButton = findViewById(R.id.calc_button);

        checkPermission(Manifest.permission.CAMERA, CAMERA_PERMISSION_CODE);
        try
        {
            classifier = new Classifier(MainActivity.this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height)
    {
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        mRGBA = new Mat(height, width, CvType.CV_8UC4);
        try
        {
            javaCameraView.turnOnTheFlash();
        }
        catch (Exception ignored)
        {

        }
    }

    @Override
    public void onCameraViewStopped()
    {
        mRGBA.release();
        mT.release();
        mRGBAT.release();
        mCROP.release();
        mTHRESH.release();
        mSKIN.release();
        hierarchy.release();
        try
        {
            javaCameraView.turnOffTheFlash();
        }
        catch (Exception x)
        {

        }
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (mRGBA != null)
            mRGBA.release();
        mRGBA = inputFrame.rgba();
        if(mSKIN != null )
            mSKIN.release();
        Rect targetFrame = new Rect((int)(mRGBA.width()*0.20), (int)(mRGBA.height()*0.10), (int)(mRGBA.width()*0.2), (int)(mRGBA.height()*0.8));
        Imgproc.rectangle(mRGBA, targetFrame.tl(), targetFrame.br(), new Scalar(0, 0, 205), 3);
        mCROP = new Mat(mRGBA.clone(), targetFrame);
        mTHRESH = new ConvertFrame().BlackWhiteFrame(mCROP); //convert frame to black/white threshold
        List<MatOfPoint> mEDGES = new ArrayList<>(); //list of detected edges
        hierarchy = new Mat();
        Imgproc.findContours(mTHRESH, mEDGES, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE); //searching for contours
        mSKIN = new Mat();
        if (!mEDGES.isEmpty() && !hierarchy.empty())
        {
            mRGBAT = PaintContoursOnFrames(mEDGES, mRGBA, new Point((int)(mRGBA.width()*0.20), (int)(mRGBA.height()*0.10)));
            mT = mRGBAT.t();
            Core.flip(mT,mSKIN,1);
            Imgproc.resize(mSKIN,mSKIN,mRGBA.size());
        }
        else
        {
            mT = mRGBA.t();
            Core.flip(mT, mSKIN, 1);
            Imgproc.resize(mSKIN,mSKIN,mRGBA.size());
        }
        return mSKIN; //zwrócenie klatki do wyświetlenia
    }

    public void onClickBtn(View view)
    {
        if(detectedContours == null || detectedContours.size() == 0)
        {
            new Handler(Looper.getMainLooper()).post(new Runnable(){
                @Override
                public void run() {textView.setText("No number detected");}});
        }
        else
        {
            tempList = detectedContours;
            StringBuilder tt = new StringBuilder();
            OperationOnNumber operation = new OperationOnNumber();
            for (int i = 0; i < tempList.size(); i++)
            {
                Mat temp = tempList.get(i).t();
                tt.append(classifier.getResult(temp));
            }
            if(tt.length() == 0)
            {
                new Handler(Looper.getMainLooper()).post(new Runnable(){
                    @Override
                    public void run() {textView.setText("No number detected");}});
            }
            else
            {
                final String txt = "Number: " + tt.toString() + System.lineSeparator() + operation.getDividers(tt.toString())
                        + System.lineSeparator() + operation.getFactors(tt.toString())
                        + System.lineSeparator() + operation.getIsOddNumber(tt.toString())
                        + System.lineSeparator() + operation.getIsPrimeNumber(tt.toString());
                new Handler(Looper.getMainLooper()).post(new Runnable(){
                    @Override
                    public void run() {textView.setText(txt);}});
            }
        }
    }

    private Mat PaintContoursOnFrames(List<MatOfPoint> listOfContours, Mat frame, Point xy)
    {
        MatOfPoint2f figures;
        Rect boundRect;
        String temp;
        Point TL, BR;
        SortMatOfPoints(listOfContours);
        Point offset = new Point(xy.x, xy.y);
        detectedContours = new ArrayList<>();
        for (int i = 0; i < listOfContours.size(); i++)
        {
            figures = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(listOfContours.get(i).toArray()), figures, 3, true);
            boundRect = Imgproc.boundingRect(new MatOfPoint(figures.toArray()));
            if ((Math.abs(boundRect.tl().x - boundRect.br().x) > (frame.height()*0.10)) && (Math.abs(boundRect.tl().x - boundRect.br().x) < ((frame.width()*0.8)/2)) &&
                    (Math.abs(boundRect.tl().y - boundRect.br().y)) > (frame.width()*0.05) && (Math.abs(boundRect.tl().y - boundRect.br().y)) < (frame.height()*0.25)) {
                t = new Mat(frame.height(), frame.width(), CvType.CV_8UC3, new Scalar(0,0,0));
                Imgproc.drawContours(t, listOfContours, i, new Scalar(255, 255, 255), -1);
                v = new Mat(t, boundRect);
                //Imgproc.erode(v,v, kernel, new Point(-1,-1),5);
                TL = new Point(boundRect.tl().x + offset.x, boundRect.tl().y + offset.y);
                BR = new Point(boundRect.br().x + offset.x, boundRect.br().y + offset.y);
                Imgproc.rectangle(frame, TL, BR, new Scalar(255, 255, 0), 3);
                detectedContours.add(v);
            }
        }
        return frame;
    }

    private void SortMatOfPoints(List<MatOfPoint> list)
    {
        List<Rect> rectangles = new ArrayList<>();
        Rect rect;
        for (int i = 0; i < list.size(); i++)
        {
            rect = Imgproc.boundingRect(list.get(i));
            rectangles.add(rect);
        }
        int unsorted = list.size() - 1;
        for (int i = 0; i < list.size() - 1; i++)
        {
            for (int j = 0; j < unsorted; j++)
            {
                if (rectangles.get(j).tl().x > rectangles.get(j+1).tl().x)
                {
                    Collections.swap(list, j, j + 1);
                }
            }
            unsorted -= 1;
        }
    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();

        if (javaCameraView != null)
        {
            try
            {
                javaCameraView.turnOffTheFlash();
            }
            catch (Exception x)
            {

            }
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        if (javaCameraView != null)
        {
            try
            {
                javaCameraView.turnOffTheFlash();
            }
            catch (Exception x)
            {

            }
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume()
    {
        super.onResume();
        try
        {
            javaCameraView.turnOnTheFlash();
        }
        catch (Exception x)
        {

        }

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