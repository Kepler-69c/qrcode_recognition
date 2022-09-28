package com.qrfinder.opencv;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    Button select;
    Bitmap bitmap;
    ImageView original_image;
    int SELECT_CODE = 12;

    Mat grad, connected;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (OpenCVLoader.initDebug()) Log.d("LOADED", "SUCCESS");
        
        select = findViewById(R.id.select);
        select.setOnClickListener(this::onClick);

    }

    private void onClick(View view) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, SELECT_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SELECT_CODE && data!=null){
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                original_image = findViewById(R.id.originalImage);
                original_image.setImageBitmap(bitmap);

                bitmap = findCode(bitmap);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    public Bitmap findCode(Bitmap bmp) {
//        initialize *******************************************************************************
        Mat mat = new Mat();
        List<MatOfPoint> contourArr = new ArrayList<>();
        List<MatOfPoint> contourArr2 = new ArrayList<>();
        List<MatOfPoint> patterns = new ArrayList<>();
        List<MatOfPoint> patternsVer1 = new ArrayList<>();
        List<MatOfPoint> patternsVer2 = new ArrayList<>();
        Utils.bitmapToMat(bmp, mat);

        Mat gray = new Mat();
        Mat img2 = new Mat();
        grad = new Mat();
        connected = new Mat();
        mat.copyTo(img2);

//        find QR code *****************************************************************************
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);

        Mat kernel1 = Mat.ones(3,3, Imgproc.MORPH_ELLIPSE);
        Imgproc.morphologyEx(gray, grad, Imgproc.MORPH_GRADIENT, kernel1);

        Imgproc.threshold(grad, grad, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 1));
        Imgproc.morphologyEx(grad, connected, Imgproc.MORPH_CLOSE, kernel2);

        Mat hierarchy = new Mat();
        Imgproc.findContours(connected, contourArr, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        //https://www.tabnine.com/code/java/methods/org.opencv.imgproc.Imgproc/minAreaRect?snippet=5ce706c17e034400044022f7
        for (int i = 0; i < contourArr.size(); i++) {
            MatOfPoint2f contour_ = new MatOfPoint2f();
            contourArr.get(i).convertTo(contour_, CvType.CV_32FC2);
            if (contour_.empty()) {continue;}

            RotatedRect rotatedRect = Imgproc.minAreaRect(contour_);
            Rect rect = Imgproc.boundingRect(contour_);

            if (rect.height > 7 && rect.width > 7 && isClose(rect.height, rect.width, (float) 0.1)) {
//                Imgproc.rectangle(img2, rect, new Scalar(0, 255, 0), 3);

//                https://stackoverflow.com/a/26026580
                Point[] vertices = new Point[4];
                rotatedRect.points(vertices);
                List<MatOfPoint> boxContours = new ArrayList<>();
                boxContours.add(new MatOfPoint(vertices));
                Imgproc.drawContours(img2, boxContours, 0, new Scalar(128, 128, 128), 3);

            } else {
                contourArr.remove(i);
                i--;
            }
        }

//        find position patterns *******************************************************************
        Mat adapThresh = new Mat();
        Mat erodeMat = new Mat();
        Mat edgesMat = new Mat();
        Mat img3 = new Mat();
        mat.copyTo(img3);
        Imgproc.adaptiveThreshold(gray,adapThresh, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 51, 0 );

        // todo: fin right kernel CV (uint8 ?)
        Mat kernel3 = Mat.ones(5,5, CvType.CV_8UC1);
        Imgproc.erode(adapThresh, erodeMat, kernel3, new Point(), 1);
        Imgproc.Canny(erodeMat, edgesMat, 50, 200);

        Mat hierarchy2 = new Mat();
        Imgproc.findContours(edgesMat, contourArr2, hierarchy2, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        for (MatOfPoint contour : contourArr2) {
            MatOfPoint2f contourFloat = toMatOfPointFloat(contour);
            double arcLen = Imgproc.arcLength(contourFloat, true) * 0.03;

            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contourFloat, approx, arcLen, true);
            Point[] points = approx.toArray();

            Log.d("SIZE", String.valueOf(approx.toArray().length));
            if (points.length == 4) {
                Log.d("SIZE_int", Arrays.toString(points));
                Rect rect = Imgproc.boundingRect(contour);
                if (rect.height > 7 && rect.width > 7 && isClose(rect.height, rect.width, (float) 0.1)) {
                    Imgproc.rectangle(img3, rect, new Scalar(36, 255, 12), 3);
                    patterns.add(contour);
//                    patterns.add(rect);
                }
            }
        }

//        verify patterns **************************************************************************
        Mat imgH = new Mat();
        Mat imgV = new Mat();
        mat.copyTo(imgH);
        mat.copyTo(imgV);

        for (MatOfPoint i : patterns) {

            Rect rect = Imgproc.boundingRect(i);
            int x = rect.x;
            int x0 = x;
            int y0 = rect.y;
            int y = y0 + rect.height/2;
            int w = rect.width;
            int h = rect.height;
            float x1, x2, x3, x4, x5, x6;
            int[] ans;
/*
            how to get the value of a pixel in an image
            double[] vec = adapThresh.get(y, x);
            int x1 = (int)vec[0];
*/
            int n = (int)adapThresh.get(y, x)[0];
            Log.d("RECT_he", String.valueOf(n));

            if((int)adapThresh.get(y, x)[0] == 255) {
                ans = arrayLoop(adapThresh, y, x0, "x", 0);
                x1 = ans[0];x0 = ans[1];
            } else {x1 = 0;}

            ans = arrayLoop(adapThresh, y, x0, "x", 255);//black   1
            x2 = ans[0];x0 = ans[1];
            ans = arrayLoop(adapThresh, y, x0, "x", 0);//white     1
            x3 = ans[0];x0 = ans[1];
            ans = arrayLoop(adapThresh, y, x0, "x", 255);//black   3
            x4 = ans[0];x0 = ans[1];
            ans = arrayLoop(adapThresh, y, x0, "x", 0);//white     1
            x5 = ans[0];x0 = ans[1];
            ans = arrayLoop(adapThresh, y, x0, "x", 255);//black   1
            x6 = ans[0];x0 = ans[1];

            boolean r1 = isClose(x2, x3, 2);
            boolean r2 = isClose(x5, x6, 2);
            boolean r3 = isClose((x2+x3)/2, x4/3, 2);
            boolean r4 = isClose((x5+x6)/2, x4/3, 2);

            Imgproc.rectangle(imgH, new Point(x, y), new Point(x + w, y), new Scalar(255, 0, 12), 3);
            Imgproc.rectangle(imgH, new Point(x, y0), new Point(x + w, y0 + h), new Scalar(36,255,12), 3);

            if (r1 && r2 && r3 && r4) {
                Imgproc.rectangle(imgV, new Point(x, y), new Point(x + w, y), new Scalar(255, 0, 12), 3);
                Imgproc.rectangle(imgV, new Point(x, y0), new Point(x + w, y0 + h), new Scalar(36,255,12), 3);
                patternsVer1.add(i);
            }
        }

        for (MatOfPoint i:patternsVer1) {
            Rect rect = Imgproc.boundingRect(i);
            int x0 = rect.x;
            int x = x0 + rect.width/2;
            int y0 = rect.y;
            int y = y0;
            int w = rect.width;
            int h = rect.height;
            float y1, y2, y3, y4, y5, y6;
            int[] ans;

            if((int)adapThresh.get(y, x)[0] == 255) {
                ans = arrayLoop(adapThresh, y, x0, "y", 0);
                y1 = ans[0];y0 = ans[1];
            } else {y1 = 0;}

            ans = arrayLoop(adapThresh, y0, x, "y", 255);//black   1
            y2 = ans[0];y0 = ans[1];
            ans = arrayLoop(adapThresh, y0, x, "y", 0);//white     1
            y3 = ans[0];y0 = ans[1];
            ans = arrayLoop(adapThresh, y0, x, "y", 255);//black   3
            y4 = ans[0];y0 = ans[1];
            ans = arrayLoop(adapThresh, y0, x, "y", 0);//white     1
            y5 = ans[0];y0 = ans[1];
            ans = arrayLoop(adapThresh, y0, x, "y", 255);//black   1
            y6 = ans[0];y0 = ans[1];

            boolean r1 = isClose(y2, y3, 2);
            boolean r2 = isClose(y5, y6, 2);
            boolean r3 = isClose((y2+y3)/2, y4/3, 2);
            boolean r4 = isClose((y5+y6)/2, y4/3, 2);

            if (r1 && r2 && r3 && r4) {
                patternsVer2.add(i);
                Imgproc.rectangle(imgV, new Point(x, y), new Point(x, y + h), new Scalar(12, 0, 255), 3);
                Imgproc.rectangle(imgV, new Point(x0, y), new Point(x0 + w, y + h), new Scalar(36,255,12), 3);
            }

        }

//        display images ***************************************************************************
        ImageView grayImg = findViewById(R.id.gray);
        Bitmap bmpGray = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(gray, bmpGray);
        grayImg.setImageBitmap(bmpGray);

        ImageView gradImg = findViewById(R.id.grad);
        Bitmap bmpGrad = Bitmap.createBitmap(grad.cols(), grad.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(grad, bmpGrad);
        gradImg.setImageBitmap(bmpGrad);

        ImageView connImg = findViewById(R.id.connected);
        Bitmap bmpConn = Bitmap.createBitmap(connected.cols(), connected.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(connected, bmpConn);
        connImg.setImageBitmap(bmpConn);

        ImageView contoursImg = findViewById(R.id.contours);
        Bitmap contoursBmp = Bitmap.createBitmap(img2.cols(), img2.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img2, contoursBmp);
        contoursImg.setImageBitmap(contoursBmp);

        ImageView adapImg = findViewById(R.id.adapThreshold);
        Bitmap adapBmp = Bitmap.createBitmap(adapThresh.cols(), adapThresh.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(adapThresh, adapBmp);
        adapImg.setImageBitmap(adapBmp);

        ImageView erodeImg = findViewById(R.id.erode);
        Bitmap erodeBmp = Bitmap.createBitmap(edgesMat.cols(), edgesMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(edgesMat, erodeBmp);
        erodeImg.setImageBitmap(erodeBmp);

        ImageView threeImg = findViewById(R.id.img3);
        Bitmap threeBmp = Bitmap.createBitmap(img3.cols(), img3.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img3, threeBmp);
        threeImg.setImageBitmap(threeBmp);

        ImageView ImgHH = findViewById(R.id.imgH);
        Bitmap BmpH = Bitmap.createBitmap(imgH.cols(), imgH.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgH, BmpH);
        ImgHH.setImageBitmap(BmpH);

        ImageView ImgVV = findViewById(R.id.imgV);
        Bitmap BmpV = Bitmap.createBitmap(imgV.cols(), imgV.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgV, BmpV);
        ImgVV.setImageBitmap(BmpV);

        return bmp;
    }

//    https://github.com/Logicify/d2g-android-client/blob/master/app/src/main/java/app/logicify/com/imageprocessing/GeomUtils.java
    public static MatOfPoint2f toMatOfPointFloat(MatOfPoint mat) {
        MatOfPoint2f matFloat = new MatOfPoint2f();
        mat.convertTo(matFloat, CvType.CV_32FC2);
        return matFloat;
    }

    private boolean isClose(float num1, float num2, float absolute) {
        float diff1 = num2/num1;
        float diff2 = num1/num2;
//        https://stackoverflow.com/a/26740680
        return Float.intBitsToFloat(Float.floatToIntBits(diff1 - diff2) & 0x7FFFFFFF) < absolute;
    }

    private static int[] arrayLoop(Mat bin, int posY, int posX, String a, int change) {
        int[] ans = new int[2];
        int length = 0;
        int ret = 0;
        while ((int)bin.get(posY, posX)[0] != change) {
            length ++;
            if (a.equals("x")) {posX ++;
            } else if (a.equals("y")) {posY ++;}
        }

        if (a.equals("x")) {ret = posX;}
        else if (a.equals("y")) {ret = posY;}
        ans[0] = length;
        ans[1] = ret;
        return ans;
    }
}