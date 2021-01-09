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
        Imgproc.threshold(blur, bwFrame, 127, 255, Imgproc.THRESH_BINARY_INV);
        return bwFrame;
    }

    public Mat BcgFrameMerger(Mat inputFrame, Mat bcgFrame)
    {
        List<Mat> t = new ArrayList<Mat>();
        t.add(inputFrame);
        t.add(inputFrame);
        t.add(inputFrame);
        t.add(new Mat(inputFrame.height(), inputFrame.width(), CvType.CV_8UC1,new Scalar(255)));
        Mat result = new Mat();
        Core.merge(t, result);
        result.copyTo(bcgFrame.submat(new Rect((int)(bcgFrame.width()*0.05), 100, (int)(bcgFrame.width()*0.9), 200)));
        return bcgFrame;
    }
}
