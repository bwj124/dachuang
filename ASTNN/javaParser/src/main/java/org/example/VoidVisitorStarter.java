package org.example;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

public class VoidVisitorStarter {
    public static String projectRoot = "D:/IDEA//Workspace/IdeaProjects/org.javaparser.examples/";
//    public static String fileRoot = projectRoot + "src/main/java/org/javasparser/examples/";
    public static String fileRoot = "";
//    public static String filePath = "src/main/java/org/javasparser/examples/ReversePolishNotation.java";

    public static void main(String[] args) throws Exception{
//        CompilationUnit cu = StaticJavaParser.parse(new File(filePath));
        List list = new JavaFileVisitor(fileRoot).getAllJavaFile();
        System.out.println(list);
//        MethodNamePrinter methodNameVisitor = new MethodNamePrinter();
//        visitAllFiles(list, methodNameVisitor);
    }

    public static void visitAllFiles(List<File> list, MethodNamePrinter mn) throws FileNotFoundException {
        for (File file : list){
            System.out.println("----------"+file.toString()+"------------");
            CompilationUnit cu = StaticJavaParser.parse(file);
            mn.visit(cu, null);
        }
    }
}
