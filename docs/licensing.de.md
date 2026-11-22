# Lizenz und Nutzung

Das Projekt steht unter der **AIhdproduction Education and Research License 1.1** vom 22.11.2026.
Der verbindliche englische Lizenztext steht in [LICENSE](../LICENSE). Diese deutsche
Zusammenfassung erklärt die Bedingungen; sie ist keine zusätzliche Lizenz.

## Rechte am Projekt

AIhdproduction behält die Rechte an den eigenen Projektbestandteilen. Die Lizenz räumt
Nutzungsrechte ein und überträgt kein Eigentum. Rechte anderer Mitwirkender sowie an Meta
DINOv3 und SAM 3, EoMT, Agisoft Metashape und anderen Abhängigkeiten bleiben bei den jeweiligen
Berechtigten.

## Private Nutzung, Schule, Lehre und Forschung: kostenlos

Für nichtkommerzielle private Nutzung, Bildung und Forschung darf die Software ohne Einzelanfrage
und ohne Lizenzgebühr genutzt, untersucht, kopiert und verändert werden.

Private Nutzung bedeutet die Nutzung durch eine Einzelperson für persönliche Zwecke, zum Beispiel
für Hobbyprojekte oder das eigene Zuhause, ohne Bezahlung und nicht für ein Unternehmen, einen
Arbeitgeber oder einen Kunden.

Zu Bildung und Forschung gehören Unterricht,
Studienarbeiten, Abschlussarbeiten und nichtkommerzielle Forschungsprojekte. Kostenlose
Weitergabe ist unter Beibehaltung der Lizenz und Hinweise erlaubt; Änderungen müssen
gekennzeichnet werden. Auch öffentlich zugängliche Quellcode-Forks sind so möglich.

Übliche Studiengebühren, Löhne, Stipendien und allgemeine Forschungsförderung machen ein
Projekt für sich genommen nicht kommerziell. Entscheidend ist der tatsächliche Zweck:
Ein bezahlter Kundenauftrag oder die Entwicklung eines kommerziellen Produkts wird nicht
allein dadurch kostenlos erlaubt, dass eine Hochschule daran arbeitet.

## Kommerzielle und jede andere Nutzung: nur nach vorheriger schriftlicher Zustimmung

Vor kommerzieller Nutzung muss AIhdproduction ausdrücklich schriftlich zustimmen.
Das betrifft insbesondere bezahlte Aufträge, Vermessungs- und Auswertungsleistungen,
betriebliche Nutzung, kommerzielle Forschung und Entwicklung sowie die Einbindung in
Produkte, APIs oder gehostete Dienste. Auch die Erzeugung von Ergebnissen für kommerzielle
Zwecke und deren spätere kommerzielle Verwertung durch den Lizenznehmer benötigen diese
Zustimmung. Auch jede andere Nutzung ausserhalb von nichtkommerzieller privater Nutzung, Bildung
und Forschung braucht die vorherige schriftliche Zustimmung von AIhdproduction.

Preis, Umfang und weitere Bedingungen werden individuell vereinbart. Eine Anfrage allein
oder eine ausbleibende Antwort ist keine Erlaubnis. Anfragen können über die
[GitHub-Issues des Projekts](https://github.com/AIhdproduction/Metashape-AI-Classifier/issues)
gestellt werden. Dort bitte keine vertraulichen Kunden- oder Projektdaten veröffentlichen.

## Eigene Daten und Ergebnisse

AIhdproduction beansprucht keine Rechte an euren Eingabedaten oder Ergebnissen allein
aufgrund der Softwarenutzung. Klassifizierte Punktwolken, Masken und Berichte werden nicht
automatisch unter dieselbe Lizenz wie der Programmcode gestellt. Bestehende Rechte an
Daten und Ergebnissen sowie an Drittmaterial bleiben zu beachten.

Ergebnisse erlaubter nichtkommerzieller privater Nutzung, Forschung und Lehre dürfen veröffentlicht
und geteilt werden, beispielsweise in Abschlussarbeiten, Publikationen und Forschungsdatensätzen.
Bei veröffentlichten Ergebnissen aus Forschung oder Lehre ist das Projekt als
„Metashape AI Classifier, AIhdproduction“ zu nennen
und nach Möglichkeit zu verlinken. Die Regel zur kommerziellen Verwertung ist eine Bedingung
der Nutzungserlaubnis für den Lizenznehmer, kein Eigentumsanspruch auf alle Ergebnisse und
keine automatisch auf jeden Empfänger übertragene Ergebnislizenz.

## Lizenzen der Modelle

Kopien beider Meta-Lizenzen liegen zur Information bei:
[SAM License für SAM 3](../licenses/SAM3_LICENSE.txt) und
[DINOv3 License für FlurSeg](../licenses/FLURSEG_DINOV3_LICENSE.md).
Sie gelten nur für die jeweiligen Modellmaterialien, nicht für den Classifier.

Beide Meta-Lizenzen erlauben unter ihren Bedingungen auch die kommerzielle Nutzung.
Die Einschränkung dieses Projekts gilt für den Classifier und nicht für die Gewichte.
Wer den Classifier kommerziell nutzen will, braucht deshalb trotzdem die vorherige schriftliche
Zustimmung von AIhdproduction.

## FlurSeg: DINOv3 License

**Built with DINOv3.** Die Gewichte von [FlurSeg](https://huggingface.co/AIhdproduction/Flurseg)
stehen unter der [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
von Meta, weil das Modell auf DINOv3 aufbaut. Die Lizenz dieses Projekts gilt nicht für die
Gewichte und ändert deren Bedingungen nicht. Die DINOv3 License verlangt unter anderem:

- bei Weitergabe der Gewichte oder davon abgeleiteter Materialien eine Kopie der Vereinbarung,
- bei Forschungspublikationen eine Erwähnung von DINOv3,
- die Einhaltung von Handelskontrollen. Bestimmte Nutzungen sind ausgeschlossen, zum Beispiel
  militärische.

Die Modellarchitektur EoMT steht unter der MIT-Lizenz, Transformers unter Apache 2.0.

## Meta SAM 3 und weitere Fremdsoftware

SAM 3 ist optional. Die Modellgewichte beider Modelle werden hier nicht mitgeliefert. Nutzer
beziehen sie separat und akzeptieren die jeweiligen Bedingungen. Der Installer installiert den
SAM-Code aus dem offiziellen Repository, auch wenn später nur FlurSeg verwendet wird.
Auch wenn anschließend nur Ergebnisse weitergegeben werden, unterliegt die Modellnutzung
weiterhin der [SAM-Lizenz von Meta](https://github.com/facebookresearch/sam3/blob/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc/LICENSE).
Bei Forschungspublikationen verlangt Meta eine Erwähnung der verwendeten SAM-Materialien.
Wer SAM-Code, Gewichte oder davon abgeleitete Materialien mitliefert, muss die entsprechenden
Weitergabebedingungen erfüllen, einschließlich einer Kopie der Meta-Lizenz.

Die App verwendet NiceGUI (MIT). Metashape benötigt eine separate Agisoft-Lizenz. Eine kommerzielle Freigabe durch
AIhdproduction ersetzt keine Genehmigung anderer Rechteinhaber. Details und Quellen stehen
in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Warum eine eigene Lizenz?

Die Lizenz bildet die gewünschte Grenze zwischen kostenloser nichtkommerzieller privater Nutzung,
Bildung und Forschung und jeder anderen, separat zu vereinbarenden Nutzung ab. Wegen dieser
Zweckbeschränkung ist das Projekt „source-available“, also mit einsehbarem Quellcode, aber
keine Open-Source-Software im Sinne der
[Open Source Definition](https://opensource.org/osd).
