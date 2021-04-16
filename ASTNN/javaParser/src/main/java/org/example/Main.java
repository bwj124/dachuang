package org.example;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

import java.io.*;

/**
 * @author bwj
 */

public class Main {
    public static String dataRoot = "F:/others/Math_slices/";
//    public static String dataRoot = "G:/dc/zyb/Math_slices_falsepositive/";

    // 遍历所有目录 Main
    // 根据slice的行号找到对应代码段 findCode -> MethodLinePrinter
    // 将代码段写入文件 MethodBodyPrinter

    public static void main(String[] args) {
        File root = new File(dataRoot);
        File[] dirs = root.listFiles();
        assert dirs != null;
        for (File dir: dirs) {
            String newDir = dir.toString();
            String codePath = "";
            String filename = "";
            String lineNum = "";
            String project = "";
            String label = "";
            File newDirFile = new File(newDir);
            File[] dataFiles = newDirFile.listFiles();
            if (dataFiles == null){
                System.out.println(dir);
                File log = new File("F:/log.txt");
                try {
                    if (!log.exists()){
                        log.createNewFile();
                    }
                    FileWriter fw = new FileWriter(log);
                    fw.write(dir.toString()+"\n");
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                continue;
            }
            for (File file: dataFiles) {
                try{
                    boolean once = true;
                    if (file.getName().endsWith(".txt")&&once){
                        String[] info = file.getName().split("\\.");
                        project = info[0];
                        filename = info[1];
                        lineNum = info[2];
                        label = info[5];
                        once = false;
                    } else {
                        codePath = file.getAbsolutePath();
                    }
                    if (file.getName().endsWith(".txt")){
                        FileReader fr = new FileReader(file);
                        BufferedReader br = new BufferedReader(fr);
                        String fileToWrite = "F:/others/allSlice/" + file.getName();
                        File ftw = new File(fileToWrite);
                        if (!ftw.exists()){
                            ftw.createNewFile();
                        }
                        FileWriter fw = new FileWriter(ftw);
                        BufferedWriter bw = new BufferedWriter(fw);
                        String oneLine;
                        while (br.ready()){
                            oneLine = br.readLine();
                            String[] lineSplit = oneLine.split("-----");
                            if (lineSplit.length > 1){
                                if (lineSplit[lineSplit.length - 1].endsWith(".java")){
                                    oneLine = lineSplit[0];
                                }else{
                                    oneLine = lineSplit[0]+lineSplit[lineSplit.length - 1];
                                }
                            }
                            bw.write(oneLine);
                            bw.newLine();
                        }
                        bw.flush();
                        bw.close();
                        br.close();
                        fw.close();
                        br.close();
                        fr.close();
                    }
                } catch (IOException e){
                    e.printStackTrace();
                }

            }

            try {
                MethodLinePrinter methodLinePrinter = new MethodLinePrinter();
                CompilationUnit compilationUnit = StaticJavaParser.parse(new File(codePath));
                methodLinePrinter.visit(compilationUnit, null);
                int i = 0;
                for (;i < methodLinePrinter.allLines.size();i ++){
                    if (Integer.valueOf(lineNum) < methodLinePrinter.allLines.get(i)) {
                        break;
                    }
                }
                int index = i - 1;
                MethodBodyPrinter methodBodyVisitor = new MethodBodyPrinter(project, filename, lineNum, label, index);
                methodBodyVisitor.visit(compilationUnit, null);
            } catch (Exception e){
                e.printStackTrace();
            }
        }
    }


}
