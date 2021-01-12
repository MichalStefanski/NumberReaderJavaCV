package com.studentproject.calculatorjavacv;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ConvertFrame
{
    public Mat BlackWhiteFrame(Mat inputFrame)
    {
        Mat grey = new Mat();
        Imgproc.cvtColor(inputFrame, grey, Imgproc.COLOR_BGR2GRAY); // klata w szarości
        // mGRAY to GaussianBlur
        Mat blur = new Mat();
        Imgproc.GaussianBlur(grey, blur, new Size(11,11),0, 0);
        Mat bwFrame = new Mat();
        Imgproc.threshold(blur, bwFrame, 127, 255, Imgproc.THRESH_BINARY);
        return bwFrame;
    }

    public Mat ScaleToModel(Mat inputFrame)
    {
        int width = inputFrame.width();
        int height = inputFrame.height();
        Rect rect;
        Mat bcg;
        Mat result = new Mat(28,28, CvType.CV_8UC(inputFrame.channels()));

        Scalar channels;
        switch (inputFrame.channels())
        {
            case 2:
                channels = new Scalar(255, 255);
                break;
            case 3:
                channels = new Scalar(255, 255, 255);
                break;
            case 4:
                channels = new Scalar(255, 255, 255, 255);
                break;
            default:
                channels = new Scalar(255);
        }
        if(width == 0 || height == 0)
        {
            return result;
        }
        if (width < height)
        {
           bcg = new Mat(height, height, CvType.CV_8UC(inputFrame.channels()), channels);
           rect = new Rect(Math.round((float)(height - width) / 2), 0, width, height);
        }
        else
        {
            bcg = new Mat(width, width, CvType.CV_8UC(inputFrame.channels()), channels);
            rect = new Rect(0, Math.round((float)(width - height) / 2), width, height);
        }
        inputFrame.copyTo(bcg.submat(rect));
        Imgproc.resize(bcg, result, result.size());
        result = new ConvertFrame().BcgFrameMerger(result);
        return result;
    }

    public Mat BcgFrameMerger(Mat inputFrame)
    {
        List<Mat> t = new ArrayList<>();
        t.add(inputFrame);
        t.add(inputFrame);
        t.add(inputFrame);
        t.add(new Mat(inputFrame.height(), inputFrame.width(), CvType.CV_8UC1,new Scalar(255)));
        Mat result = new Mat();
        Core.merge(t, result);
        //result.copyTo(bcgFrame.submat(new Rect((int)(bcgFrame.width()*0.05), 100, (int)(bcgFrame.width()*0.9), 200)));
        return result;
    }
}
