package org.example;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

public class RootFileGetter {
//    public String sourceRoot = "F:/others/Math_slices/";
    public String file = "";
    public String fileLine = "";
    public JavaFileVisitor javaFileVisitor = null;
    public File root = null;

    public RootFileGetter(String project, String fileName, String fileLine, String label,File root) throws FileNotFoundException {
        this.file = fileName;
        this.root = root;
        this.fileLine = fileLine;
        this.function(project, this.file, this.fileLine, label, this.root);
    }

    public void function(String project, String filename, String fileLine, String label, File dir) throws FileNotFoundException {
        File[]files=dir.listFiles();
        for(File file:files)
        {
            if(file.isDirectory())
            {
                function(project, filename, fileLine, label, file.getAbsoluteFile());
            }
            if(file.isFile() && filename.equals(file.getName()))
            {
//                System.out.println("要查找的文件路径为："+file.getAbsolutePath());
//                javaFileVisitor = new JavaFileVisitor(file.getParent()+"/");
//                List<File> list = this.javaFileVisitor.getAllJavaFile();

//                MethodNamePrinter methodNameVisitor = new MethodNamePrinter(file.getName().replace(".java", ""));
                MethodLinePrinter methodLinePrinter = new MethodLinePrinter();
                CompilationUnit compilationUnit = StaticJavaParser.parse(file);
                methodLinePrinter.visit(compilationUnit, null);
                int i = 0;
                for (;i < methodLinePrinter.allLines.size();i ++){
                    if (Integer.valueOf(fileLine) < methodLinePrinter.allLines.get(i)) {
                        break;
                    }
                }
                int index = i - 1;

                MethodBodyPrinter methodBodyVisitor = new MethodBodyPrinter(project, file.getName().replace(".java", ""), fileLine, label, index);
                methodBodyVisitor.visit(compilationUnit, null);

//                visitAllFiles(list, methodNameVisitor);
//                visitFile(file, methodBodyVisitor);
                break;
            }
        }
        return;
    }

    public void visitAllFiles(List<File> list, MethodNamePrinter mn) throws FileNotFoundException {
        for (File file : list){
            System.out.println("----------"+file.toString()+"------------");
            CompilationUnit cu = StaticJavaParser.parse(file);
            mn.visit(cu, null);
        }
    }

    public void visitFile(File filename, MethodBodyPrinter mn) throws FileNotFoundException {
        System.out.println("----------"+file.toString()+"------------");
        CompilationUnit cu = StaticJavaParser.parse(filename);
        mn.visit(cu, null);
    }
}

