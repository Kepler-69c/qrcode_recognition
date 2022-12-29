package com.qrfinder.opencv;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;

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
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    Button select;
    Bitmap bitmap;

    Mat grad, connected;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (OpenCVLoader.initDebug()) Log.d("LOADED OpenCV", "SUCCESS");

        select = findViewById(R.id.select);
        select.setOnClickListener(this::onClick);

    }

    private void onClick(View view) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");

        launchSomeActivity.launch(intent);
    }

    ActivityResultLauncher<Intent> launchSomeActivity
            = registerForActivityResult(
            new ActivityResultContracts
                    .StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null && data.getData() != null) {
                        Uri selectedImageUri = data.getData();
                        try {
                            if (OpenCVLoader.initDebug()) Log.d("LOADED Image", "SUCCESS");

                            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);

                            clearView();
                            displayBitmap(bitmap);

                            Map.Entry<Boolean, Bitmap> returnEntry = findCode(bitmap);
                            Boolean qrFound = returnEntry.getKey();
                            Bitmap qrBmp = returnEntry.getValue();

                            if (qrFound) {
                                displayBitmap(qrBmp);
                            }

                        }
                        catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });

    public class qrCode {
//        https://stackoverflow.com/a/18341560
        private final boolean confirmed;
        private Mat initial;
        private Bitmap bmp;

        private Mat makeGray(Mat mat) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
            return mat;
        }

        private Mat aThresh(Mat mat) {
            Imgproc.adaptiveThreshold(mat, mat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 51, 0);
            return mat;
        }

        public qrCode(Mat initial, Mat cut, MatOfPoint code) {
            Mat gray = makeGray(cut);
            Mat adapThresh = aThresh(gray);
            List<MatOfPoint> nPatterns = findPatterns(cut, adapThresh);
            List<MatOfPoint> vPatterns = verifyPatters(initial, adapThresh, nPatterns);
            confirmed = verifyCode(code, vPatterns);

            if (confirmed) {
                displayMat(cut);
            }
        }
    }


    public Map.Entry<Boolean,Bitmap> findCode(Bitmap bmp) {
//        initialize ********************************************************https://stackoverflow.com/a/46812543***********************
        Mat mat = new Mat();
        Utils.bitmapToMat(bmp, mat);

        List<MatOfPoint> qrArr = new ArrayList<>();
        boolean qrFound = false;

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
        Imgproc.findContours(connected, qrArr, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        //https://www.tabnine.com/code/java/methods/org.opencv.imgproc.Imgproc/minAreaRect?snippet=5ce706c17e034400044022f7
        for (int i = 0; i < qrArr.size(); i++) {
            MatOfPoint2f contour_ = new MatOfPoint2f();
            qrArr.get(i).convertTo(contour_, CvType.CV_32FC2);
            if (contour_.empty()) {continue;}

            Rect rect = Imgproc.boundingRect(contour_);

            if (rect.height > 7 && rect.width > 7 && isClose(rect.height, rect.width, (float) 0.1)) {
                Mat cut = cutAndTransform(mat, contour_);
                img2 = cutAndTransform(img2, contour_);
                MatOfPoint code = qrArr.get(i);
                qrCode qrcode = new qrCode(img2, cut, code);

                if (qrcode.confirmed) {
                    Bitmap cutBmp = Bitmap.createBitmap(cut.cols(), cut.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(cut, cutBmp);

                    bmp = cutBmp;
                    qrFound = true;

                    break;

                }
            }
        }

//        https://stackoverflow.com/a/46812543
        AbstractMap.SimpleEntry<Boolean, Bitmap> simpleEntry = new AbstractMap.SimpleEntry<>(qrFound, bmp);
        return simpleEntry;
    }

    private void displayMat(Mat mat) {
//        initialising new layout
        ImageView imageView = new ImageView(MainActivity.this);
//        setContentView(R.layout.activity_main);
        LinearLayout layout = (LinearLayout)findViewById(R.id.imgLayout);
        LinearLayout.LayoutParams params = new LinearLayout
                .LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);

//        create bitmap and add to View
        Bitmap bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);//Bitmap.Config.RGB_565
        Utils.matToBitmap(mat, bmp);
        imageView.setImageBitmap(bmp);
        imageView.setLayoutParams(params);
        layout.addView(imageView);
    }

    private void displayBitmap(Bitmap bmp) {
        //        initialising new layout
        ImageView imageView = new ImageView(MainActivity.this);
//        setContentView(R.layout.activity_main);
        LinearLayout layout = (LinearLayout)findViewById(R.id.imgLayout);
        LinearLayout.LayoutParams params = new LinearLayout
                .LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);

//        create bitmap and add to View
        imageView.setImageBitmap(bmp);
        imageView.setLayoutParams(params);
        layout.addView(imageView);
    }

    private void clearView() {
        LinearLayout layout = (LinearLayout) findViewById(R.id.imgLayout);
        layout.removeAllViews();
    }

    private static List<MatOfPoint> findPatterns(Mat mat, Mat adapThresh) {
        List<MatOfPoint> contourArr2 = new ArrayList<>();
        List<MatOfPoint> patterns = new ArrayList<>();
        Mat erodeMat = new Mat();
        Mat edgesMat = new Mat();
        Mat img3 = new Mat();
        mat.copyTo(img3);

        //        https://stackoverflow.com/a/14071387
        int u = 0;
        Mat kernel3 = Mat.ones(5, 5, u);
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
                }
            }
        }

        return patterns;
    }

    private List<MatOfPoint> verifyPatters(Mat mat, Mat adapThresh, List<MatOfPoint> patterns) {
        List<MatOfPoint> patternsVer1 = new ArrayList<>();
        List<MatOfPoint> patternsVer2 = new ArrayList<>();

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
             -> how to get the value of a pixel in a Mat:
            double[] vec = adapThresh.get(y, x);
            int x1 = (int)vec[0];

            -> shorter:
            int n = (int)adapThresh.get(y, x)[0];
            Log.d("RECT_he", String.valueOf(n));
*/

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
            int y = rect.y;
            int y0 = y;
            int w = rect.width;
            int h = rect.height;
            float y1, y2, y3, y4, y5, y6;
            int[] ans;

            if((int)adapThresh.get(y, x)[0] == 255) {
                ans = arrayLoop(adapThresh, y0, x, "y", 0);
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

        displayMat(imgH);
        displayMat(imgV);

        return patternsVer2;
    }

    private static Boolean verifyCode(MatOfPoint code, List<MatOfPoint> patterns) {
        boolean verified = false;
        Rect QrRect = Imgproc.boundingRect(code);

        for (MatOfPoint j:patterns) {
            Rect PatternRect = Imgproc.boundingRect(j);
            int xp = PatternRect.x + PatternRect.width/2;
            int yp = PatternRect.y + PatternRect.height/2;

//                java.avt.Polygon();
            boolean contain = QrRect.contains(new Point(xp, yp));
            if (contain) {
                verified = true;
            }
        }

        return verified;
//      TODO: Imgproc.rectangle -> show qr Code
    }

    private static Mat cutAndTransform(Mat mat, MatOfPoint2f contour_) {
        Mat qrImg = new Mat(400, 400, mat.type());

        RotatedRect rotatedRect = Imgproc.minAreaRect(contour_);
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
//        https://stackoverflow.com/a/36058630
        Mat src = new MatOfPoint2f(vertices);
        Mat dst = new MatOfPoint2f(new Point(0, 0), new Point(qrImg.width() - 1, 0), new Point(qrImg.width() - 1, qrImg.height() - 1), new Point(0, qrImg.height() - 1));

//        TODO: replace built-in function
        Mat transform = Imgproc.getPerspectiveTransform(src, dst);
        Imgproc.warpPerspective(mat, qrImg, transform, qrImg.size());

        return qrImg;
    }

    //    https://github.com/Logicify/d2g-android-client/blob/master/app/src/main/java/app/logicify/com/imageprocessing/GeomUtils.java
    private static MatOfPoint2f toMatOfPointFloat(MatOfPoint mat) {
        MatOfPoint2f matFloat = new MatOfPoint2f();
        mat.convertTo(matFloat, CvType.CV_32FC2);
        return matFloat;
    }

    private static boolean isClose(float num1, float num2, float absolute) {
        float diff1 = num2/num1;
        float diff2 = num1/num2;
//        https://stackoverflow.com/a/26740680
        return Float.intBitsToFloat(Float.floatToIntBits(diff1 - diff2) & 0x7FFFFFFF) < absolute;
    }

    private static int[] arrayLoop(Mat bin, int posY, int posX, String a, int change) {
        int[] ans = new int[2];
        int length = 0;
        int ret = 0;
        // TODO: somehow sometimes the Mat is 0 -> nullPointerException, but I don't have time now to figure out why (DEC 12)
        try {
            while ((int)bin.get(posY, posX)[0] != change) {
                length ++;
                if (a.equals("x")) {posX ++;
                } else if (a.equals("y")) {posY ++;}
            }
        } catch (Exception e) {e.printStackTrace();}

        if (a.equals("x")) {ret = posX;}
        else if (a.equals("y")) {ret = posY;}
        ans[0] = length;
        ans[1] = ret;
        return ans;
    }
}