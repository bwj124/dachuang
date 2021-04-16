package org.example;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;


public class MethodNamePrinter extends VoidVisitorAdapter {
    public int count = 0;
    public File srcFile = null;
    public File labelsFile = null;

    public MethodNamePrinter(String fileName){
        int i = 0;
        String projectRoot = "F:/大创/Bug Detection/ASTNN/TransASTNN/simfix_supervised_data/mine/";
        File tmpDir = new File(projectRoot+i);
        for (;tmpDir.exists();){
            i ++;
            tmpDir = new File(projectRoot+i);
        }
        tmpDir.mkdir();
        this.srcFile = new File(projectRoot+i+"/src.tsv");
        this.labelsFile = new File(projectRoot+i+"/lables.csv");

        File f = new File(projectRoot+i+"/fileName");
        try {
            f.createNewFile();
            OutputStreamWriter fo = new OutputStreamWriter(new FileOutputStream(f));
            fo.write(fileName);
            fo.close();

            this.labelsFile.createNewFile();
            FileOutputStream los = new FileOutputStream(this.labelsFile);
            OutputStreamWriter losw = new OutputStreamWriter(los, "UTF-8");
            losw.write("id2,label\n");
            losw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void visit(MethodDeclaration md, Object arg) {
        super.visit(md, arg);
        md.removeJavaDocComment();
        count ++;
        System.out.println(count+"\t\""+md.removeComment().toString().replace("\"", "\"\"")+"\"");
        String str = count+"\t\""+md.removeComment().toString().replace("\"", "\"\"")+"\"";

        try{
            FileOutputStream fos = null;
            FileOutputStream los = null;
            if(!this.srcFile.exists()){
                this.srcFile.createNewFile();
                fos = new FileOutputStream(this.srcFile);
            }else{
                //如果文件已存在，那么就在文件末尾追加写入
                fos = new FileOutputStream(this.srcFile,true);
                OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");
                osw.write(str+"\n");
                osw.close();
            }

            FileOutputStream nlos = new FileOutputStream(this.labelsFile,true);
            OutputStreamWriter losw = new OutputStreamWriter(nlos, "UTF-8");
            losw.write(count+",1"+"\n");
            losw.close();

        } catch (Exception e){
            e.printStackTrace();
        }

    }


}
