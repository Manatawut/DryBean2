library(shiny)
library(readxl)
library(e1071)
library(ggplot2)
library(caret)
library(markdown)
library(lattice)
library(caTools)
library(DT)

df <-read_excel("Dry_Bean_Dataset.xlsx")
df$Class <- factor(df$Class)
vars <- setdiff(names(df), "Class")
selectKernel <- c("linear","radial")
selectModel <- c("K-Nearest Neighbors"="KNN","Support Vector Machine (SVM)"="SVM")
ui <- fluidPage(
    navbarPage(
        "DryBean",
        tabPanel("About Data",
                 includeMarkdown("about.md")),
        tabPanel("Exploratory Data Analysis (EDA)",
                 mainPanel(
                     tabsetPanel(
                         tabPanel("Overview",
                                  br(),
                                  h4("Data Preview"),
                                  tableOutput("head"),
                                  h4("Data Summary"),
                                  verbatimTextOutput("summary")),
                         tabPanel("Plot",
                                  br(),
                                  h4("Please Select Variable: "),
                                  selectInput(inputId = 'predic1',label = 'Variable (X)',vars),
                                  selectInput(inputId = 'predic2',label = 'Variable (Y)',vars, selected = vars[2]),
                                  plotOutput('plot'))
                     )
                 )
        ),
        tabPanel("Model",
                 sidebarPanel(
                     selectInput(inputId = 'model',label = 'Please Select Model: ',selectModel),
                     conditionalPanel(condition = "input.model == 'KNN'",
                                      sliderInput(inputId = 'knum',label = "Select k",3,min = 1,max=20)),
                     conditionalPanel(condition = "input.model == 'SVM'",
                                      selectInput(inputId = 'kernel',label = 'Select Kernel Type',selectKernel),
                                      sliderInput(inputId = 'cross',label = "Select Number of Cross-Validation",5,min = 1,max=10),
                                      sliderInput(inputId = 'cost',label = "Select Cost",1,min = 1,max=10),
                                      sliderInput(inputId = 'gamma',label = "Select Gamma",1,min = 1,max=10)
                     )
                 ),
                 mainPanel(
                     tabsetPanel(
                         tabPanel("Data Preview",
                                  h4("Data for Training Model"),
                                  DT::dataTableOutput("datatrain"),
                                  h4("Data for test Model"),
                                  DT::dataTableOutput("datatest")
                                  ),
                         tabPanel("Summary",
                                  conditionalPanel(condition = "input.model == 'KNN'",
                                                   h4("Confusion Matrix"),
                                                   verbatimTextOutput('confusknn'),
                                                   h4("Accuracy"),
                                                   verbatimTextOutput('predictionknn')
                                                   ),
                                  conditionalPanel(condition = "input.model == 'SVM'",
                                                   h4("Summary Model"),
                                                   verbatimTextOutput('summarysvm'),
                                                   h4("Confusion Matrix"),
                                                   verbatimTextOutput('confussvm'),
                                                   h4("Accuracy"),
                                                   verbatimTextOutput('predictionsvm')
                                                   )
                                  )
                         )
                     )
                 )
        
    )
)
server <- function(input, output) {
    set.seed(1234)
    train.id <- sample.split(df$Class, SplitRatio = 0.70)
    df.train <- subset(df, train.id) 
    df.test <- subset(df, !train.id)
    
    modelknn <- reactive({
        model = knn3(data=df.train,Class~.,k=input$knum)
    })
    
    modelsvm <- reactive({
        model.lm <-
            svm(
                Class ~.,
                kernel = input$kernel,
                cross = input$cross,
                scale = F,
                data = df.train,
                cost = input$cost,
                gamma = input$gamma
            )
    })
    
    output$datatrain <- DT::renderDataTable({
        DT::datatable(df.train, options = list(pageLength = 10))
    })
    
    output$datatest <- DT::renderDataTable({
        DT::datatable(df.test, options = list(pageLength = 10))
    })
    
    output$confusknn <- renderPrint({
        predicted  = predict(modelknn(), df.test, type = "class")
        conf.table <- table(predicted, df.test$Class)
        conf.table
    })
    
    output$predictionknn <- renderPrint({
        predicted  = predict(modelknn(), df.test, type = "class")
        conf.table <- table(predicted, df.test$Class)
        ACC <- ( (conf.table[1,1]+conf.table[2,2]+conf.table[3,3]+conf.table[4,4]+conf.table[5,5]+conf.table[6,6]+conf.table[7,7]) / sum(conf.table) ) * 100
        ACC
    })
    
    output$summarysvm <- renderPrint({
        summary(modelsvm())
    })
    
    output$confussvm <- renderPrint({
        pred <- predict(modelsvm(), newdata = df.test)
        conf.table <- table(predicted = pred, actual = df.test$Class)
        conf.table
    })
    
    output$predictionsvm <- renderPrint({
        pred <- predict(modelsvm(), newdata = df.test)
        conf.table <- table(predicted = pred, actual = df.test$Class)
        acc = sum(pred == df.test$Class) / length(pred) * 100
        acc
    })
    
    output$head <- renderTable({
        head(df,10)})
    
    output$summary <- renderPrint({
        summary(df)})
    
    output$plot <- renderPlot({
        ggplot(df, aes_string(x = input$predic1, y = input$predic2)) +
            geom_point(aes(color = Class), size = 1, alpha = .7) + 
            theme_bw()
    })
    
    
}
shinyApp(ui = ui, server = server)
