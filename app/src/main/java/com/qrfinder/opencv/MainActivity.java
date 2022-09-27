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
import java.util.Iterator;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    Button select;
    Bitmap bitmap;
    ImageView original_image;
    int SELECT_CODE = 12;

    Mat grad, connected, thresh;

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
        Mat mat = new Mat();
        List<MatOfPoint> contourArr = new ArrayList<>();
        Utils.bitmapToMat(bmp, mat);

        Mat gray = new Mat();
        Mat img2 = new Mat();
        grad = new Mat();
        connected = new Mat();
        mat.copyTo(img2);

        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);

        Mat kernel1 = Mat.ones(3,3, Imgproc.MORPH_ELLIPSE);
        Imgproc.morphologyEx(gray, grad, Imgproc.MORPH_GRADIENT, kernel1);

        Imgproc.threshold(grad, grad, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 1));
        Imgproc.morphologyEx(grad, connected, Imgproc.MORPH_CLOSE, kernel2);

        Mat hierarchy = new Mat();
        Imgproc.findContours(connected, contourArr, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

//        Iterator<MatOfPoint> each = contourArr.iterator();
//        each = contourArr.iterator();
//        while (each.hasNext()) {
//            MatOfPoint contour = each.next();
//            Rect rect = Imgproc.boundingRect(contour);
////            Imgproc.minAreaRect(img2, rect);
//            if (rect.height < 7 && rect.width < 7) {
//                Imgproc.rectangle(img2, rect.tl(), rect.br(), new Scalar(255,
//                        0, 0), 1);
//            }
//        }

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

        return bmp;
    }

    private boolean isClose(float height, float width, float absolute) {
        float diff1 = width/height;
        float diff2 = height/width;
        return Float.intBitsToFloat(Float.floatToIntBits(diff1 - diff2) & 0x7FFFFFFF) < absolute;
    }
}