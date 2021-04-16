package org.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author bwj
 */
public class App {
    //public static String dataRoot = "F:/others/Math_slices/";
    public static String dataRoot = "G:/dc/zyb/Math_slices_falsepositive/";
    public static String ggnnDataRoot = "F:/others/ggnn_vector/";

    public static void main( String[] args ) throws FileNotFoundException {
        File root = new File(dataRoot);

        List<List> list = getFileName();
        for (int i = 0;i < list.get(0).size();i ++) {
            String file_name = list.get(0).get(i).toString();
            String file_line = list.get(1).get(i).toString();
            RootFileGetter rootFileGetter = new RootFileGetter(file_name, file_line, root);
        }

    }

    public static List<List> getFileName(){
        File root = new File(ggnnDataRoot);
        String[] files = root.list();
        List<List> list = new ArrayList<>(2);
        List<String> fileNames = new ArrayList<>();
        List<String> fileLines = new ArrayList<>();
        String tmp_name = "";
        String tmp_line = "";
        for (String str:files) {
            System.out.println(str);
            String[] tmp_list = str.split("\\.");
            tmp_name = tmp_list[1] +".java";
            tmp_line = tmp_list[2];
            fileNames.add(tmp_name);
            fileLines.add(tmp_line);
        }

//        Set set_name = new HashSet(fileNames);
//        fileNames = new ArrayList<>(set_name);
//        Set set_line = new HashSet(fileLines);
//        fileLines = new ArrayList<>(set_line);
        list.add(fileNames);
        list.add(fileLines);

        return list;
    }
}
