package org.example;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.*;


public class MethodBodyPrinter extends VoidVisitorAdapter {
    public int count = 0;
    public File srcFile = null;
    public File labelsFile = null;
    public File logFile = null;
    public int index;
    public int num = 0;
    public String fileName;
    public String fileLine;
    public String project;
    public String label;

    public MethodBodyPrinter(String project, String filename, String fileLine, String label, int index) {
        String projectRoot = "F:/大创/Bug Detection/ASTNN/TransASTNN/simfix_supervised_data/mine/555/";
        this.fileName = filename;
        this.fileLine = fileLine;
        this.project = project;
        this.label = label;

        this.srcFile = new File(projectRoot+"src.tsv");
        this.labelsFile = new File(projectRoot+"lables.csv");
        this.logFile = new File(projectRoot+"log");
        this.index = index;

        try {
            if (!this.labelsFile.exists()){
                this.labelsFile.createNewFile();
                FileOutputStream los = new FileOutputStream(this.labelsFile);
                OutputStreamWriter losw = new OutputStreamWriter(los, "UTF-8");
                losw.write("id2,label\n");
                losw.close();
            }else{
                String lastLine = readLastLine(this.labelsFile, "UTF-8");
                System.out.println(lastLine);
                this.num = Integer.valueOf(lastLine.split(",")[0]);
            }

            if (!this.srcFile.exists()){
                this.srcFile.createNewFile();
            }

            if (!this.logFile.exists()){
                this.logFile.createNewFile();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void visit(MethodDeclaration md, Object arg) {
        super.visit(md, arg);
        if (this.count != this.index){
            count ++;
        } else {
            md.removeJavaDocComment();
            count ++;
            int num_tmp = num + 1;
            System.out.println(num_tmp+"\t\""+md.removeComment().toString().replace("\"", "\"\"")+"\"");
            String str = num_tmp+"\t\""+md.removeComment().toString().replace("\"", "\"\"")+"\"";

            try{
                FileOutputStream fos = null;
                FileOutputStream los = null;
                FileOutputStream log = null;
                fos = new FileOutputStream(this.srcFile,true);
                OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");
                osw.write(str+"\n");
                osw.close();

                los = new FileOutputStream(this.labelsFile,true);
                OutputStreamWriter losw = new OutputStreamWriter(los, "UTF-8");
                losw.write(num_tmp+",1"+"\n");
                losw.close();

                log = new FileOutputStream(this.logFile, true);
                OutputStreamWriter logw = new OutputStreamWriter(log, "UTF-8");
                logw.write(this.project+"."+this.fileName+"."+this.fileLine+"."+this.label+"\n");
                logw.close();
            } catch (Exception e){
                e.printStackTrace();
            }


        }
    }

    public String readLastLine(File file, String charset) throws IOException {
        if (!file.exists() || file.isDirectory() || !file.canRead()) {
            return null;
        }
        RandomAccessFile raf = null;
        try {
            raf = new RandomAccessFile(file, "r");
            long len = raf.length();
            if (len == 0L) {
                return "";
            } else {
                long pos = len - 1;
                while (pos > 0) {
                    pos--;
                    raf.seek(pos);
                    if (raf.readByte() == '\n') {
                        break;
                    }
                }
                if (pos == 0) {
                    raf.seek(0);
                }
                byte[] bytes = new byte[(int) (len - pos)];
                raf.read(bytes);
                if (charset == null) {
                    return new String(bytes);
                } else {
                    return new String(bytes, charset);
                }
            }
        } catch (FileNotFoundException e) {
        } finally {
            if (raf != null) {
                try {
                    raf.close();
                } catch (Exception e2) {
                }
            }
        }
        return null;
    }
}
