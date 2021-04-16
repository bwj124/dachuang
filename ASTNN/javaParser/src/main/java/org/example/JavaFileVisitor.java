package org.example;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * @author bwj
 */
public class JavaFileVisitor {
    String dirPath = "";

    public JavaFileVisitor(String dirPath){
        this.dirPath = dirPath;
    }

    public List<File> getAllJavaFile(){
        int fileNum = 0, folderNum = 0;
        File root = new File(this.dirPath);
        ArrayList<File> list = new ArrayList<>();

        if (root.exists()){
            if (root.listFiles() == null){
                return null;
            }
            File[] files = root.listFiles();
            for (File file : files){
                if (file.isFile()){
                    if (isJavaFile(file.toString())){
                        list.add(file);
                        fileNum ++;
                    } 
                }
            }
        } else {
            System.out.println("文件不存在!");
        }
        System.out.println("文件夹数量:" + folderNum + ",java文件数量:" + fileNum);
        return list;
    }

    public boolean isJavaFile(String str){
        return Pattern.matches(".*\\.java", str);
    }

}
