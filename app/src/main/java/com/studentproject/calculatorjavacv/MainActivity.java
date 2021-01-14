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
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.imgproc.Imgproc;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.view.View;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2
{

    JavaCameraView javaCameraView;
    Mat mRGBA, mRGBAT, mCROP, mTHRESH, mSKIN, mT, hierarchy;
    TextView textView;
    Button calcButton;
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final String MODEL_PATH = "mnist.tflite";
    Classifier classifier;

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(MainActivity.this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case BaseLoaderCallback.SUCCESS:
                {
                    javaCameraView.enableView();
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

        javaCameraView = findViewById(R.id.basic_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(MainActivity.this);

        textView = findViewById(R.id.predicted_text);
        calcButton = findViewById(R.id.calc_button);

        calcButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

            }
        });

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
        mRGBA = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped()
    {
        mRGBA.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRGBA = inputFrame.rgba();
        //draw rectangle on preview
        Rect targetFrame = new Rect((int)(mRGBA.width()*0.10), (int)(mRGBA.height()*0.20), (int)(mRGBA.width()*0.8), (int)(mRGBA.height()*0.20));
        Imgproc.rectangle(mRGBA, targetFrame.tl(), targetFrame.br(), new Scalar(0, 0, 205), 3);
        //crop frame to rectangle size
        mCROP = new Mat(mRGBA.clone(), targetFrame);
        //mTHRESH to list of contours
        mTHRESH = new ConvertFrame().BlackWhiteFrame(mCROP); //convert frame to black/white threshold
        List<MatOfPoint> mEDGES = new ArrayList<>(); //list of detected edges
        hierarchy = new Mat();
        Imgproc.findContours(mTHRESH, mEDGES, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE); //searching for contours
        mSKIN = new Mat();
        if (!mEDGES.isEmpty() && !hierarchy.empty())
        {
            mRGBAT = markContoursOnFrame(mEDGES, mRGBA);
            mT = mRGBAT.t();
            Core.flip(mT,mSKIN,1);
            Imgproc.resize(mSKIN,mSKIN,mRGBA.size()); // przeskalowanie klatki
            return mSKIN; //zwrócenie klatki do wyświetlenia
        }
        else
        {
            mT = mRGBA.t();
            Core.flip(mT, mSKIN, 1);
            Imgproc.resize(mSKIN,mSKIN,mRGBA.size());
            return mSKIN;
        }
    }

    private Mat markContoursOnFrame(List<MatOfPoint> listOfContours, Mat zerosFrame) {
        Mat frame = zerosFrame.clone();
        Point offset = new Point((int)(frame.width()*0.10), (int)(frame.height()*0.20));
        MatOfPoint2f figures;
        List<Rect> boundRects = new ArrayList<>();
        String tt = "";

        Rect boundRect ;
        for (int i = 0; i < listOfContours.size(); i++) {
            figures = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(listOfContours.get(i).toArray()), figures, 3, true);
            boundRect = Imgproc.boundingRect(new MatOfPoint(figures.toArray()));
            try
            {
                if ((Math.abs(boundRect.tl().x - boundRect.br().x) > 20) && (Math.abs(boundRect.tl().x - boundRect.br().x) < Math.round(frame.width()/2)) && (Math.abs(boundRect.tl().y - boundRect.br().y)) > 20 && (Math.abs(boundRect.tl().y - boundRect.br().y)) < frame.height())
                {
                    boundRects.add(boundRect);
                }
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
        }

        for (Rect rect: boundRects)
        {
            String t;
            Point TL = new Point(rect.tl().x + offset.x, rect.tl().y + offset.y);
            Point BR = new Point(rect.br().x + offset.x, rect.br().y + offset.y);
            Imgproc.rectangle(frame, TL, BR, new Scalar(255, 255, 0), 3);
            t = classifier.getResult(new Mat(new ConvertFrame().BlackWhiteFrame(frame.clone()), rect));
            tt += t;
        }
        return frame;
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